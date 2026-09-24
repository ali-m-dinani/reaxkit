"""Scientific and failure-path gates for the unified runtime rollout."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_info

from reaxkit.analysis.connectivity.connectivity import (
    BondEventsRequest, BondEventsTask, ConnectionStatsRequest, ConnectionStatsTask,
)
from reaxkit.analysis.connectivity.stream_state import BondTraceState
from reaxkit.analysis.timeseries.timeseries import TrajectoryCoordinateSeriesTask, TrajectoryCoordinateSeriesRequest
from reaxkit.analysis.timeseries.timeseries import TrajectoryDisplacementSeriesTask, TrajectoryDisplacementSeriesRequest
from reaxkit.analysis.trajectory.dihedral import DihedralTask, DihedralRequest
from reaxkit.analysis.trajectory.msd import MSDTask, MSDRequest
from reaxkit.core.runtime.artifacts import ArtifactSpec, ArtifactWriter
from reaxkit.core.runtime.execution_contracts import TaskCapabilities, ExecutionShape, resolve_execution_policy
from reaxkit.core.runtime.frame_pipeline import BoundedFramePipeline
from reaxkit.domain.data_models import TrajectoryData, SimulationData, ConnectivityData
from reaxkit.utils.numerical.signal_ops import schmitt_hysteresis, clean_flicker


def trajectory():
    rng = np.random.default_rng(812)
    positions = np.cumsum(rng.normal(size=(13, 4, 3)), axis=0) % 10
    sim = SimulationData(atom_ids=[2, 5, 8, 11], cell_lengths=np.full((13, 3), 10.0),
                         cell_angles=np.full((13, 3), 90.0), iterations=np.arange(13) * 10)
    return TrajectoryData(positions=positions, atom_ids=[2, 5, 8, 11], elements=["C", "N", "C", "H"],
                          iterations=sim.iterations, simulation=sim)


def frame_stream(data):
    for index in range(len(data.positions)):
        sim = replace(data.simulation, cell_lengths=data.simulation.cell_lengths[index:index+1],
                      cell_angles=data.simulation.cell_angles[index:index+1], iterations=data.iterations[index:index+1])
        yield replace(data, positions=data.positions[index:index+1], iterations=data.iterations[index:index+1],
                      source_frame_indices=np.array([index]), simulation=sim)


def pipeline(task, request, workers=3, chunk=4):
    return BoundedFramePipeline(resolve_execution_policy(task, request, {"workers": workers, "chunk_size": chunk},
                                                         environ={"SLURM_CPUS_PER_TASK": "4"}))


@pytest.mark.parametrize("task,settings", [
    (TrajectoryCoordinateSeriesTask(), TrajectoryCoordinateSeriesRequest(frames=[1, 3, 6, 9], every=2, atom_ids=[5, 11])),
    (DihedralTask(), DihedralRequest(frames=[1, 3, 6, 9], every=2, atom_ids=[2, 5, 8, 11])),
])
def test_frame_kernels_match_frozen_serial_path(task, settings):
    request = settings
    data = trajectory()
    expected = task.run(data, request).table
    for workers in (1, 3):
        runtime = pipeline(task, request, workers)
        actual = task.run_stream(frame_stream(data), request, pipeline=runtime).table
        pd.testing.assert_frame_equal(actual, expected)
        assert runtime.metrics.peak_in_flight <= 4


@pytest.mark.parametrize("unwrap", [False, True])
@pytest.mark.parametrize("chunk", [1, 3, 8])
def test_blocked_msd_matches_all_origins_and_periodic_images(unwrap, chunk):
    data = trajectory()
    data.simulation.cell_lengths[:, 0] += np.arange(13) * 0.1
    data.simulation.cell_angles[:, 2] = 75
    request = MSDRequest(frames=[0, 2, 4, 6, 9, 12], dims=("x", "z"), atom_ids=[11, 5], unwrap=unwrap)
    task = MSDTask()
    actual = task.run_blocks(frame_stream(data), request, pipeline=pipeline(task, request, chunk=chunk)).table
    pd.testing.assert_frame_equal(actual, task.run(data, request).table, atol=1e-12, rtol=1e-12)


def connectivity():
    rng = np.random.default_rng(34)
    bo = rng.random((19, 3, 3))
    return ConnectivityData(bond_orders=bo, atom_ids=[2, 5, 7], elements=["C", "N", "H"], iterations=np.arange(19)*10)


def connectivity_frames(data):
    for i in range(len(data.bond_orders)):
        yield replace(data, bond_orders=data.bond_orders[i:i+1], iterations=data.iterations[i:i+1], source_frame_indices=np.array([i]))


@pytest.mark.parametrize("how", ["count", "mean", "max"])
def test_connection_reducer_matches_serial_grouping(how):
    data = connectivity()
    request = ConnectionStatsRequest(how=how, frames=[0, 2, 4, 8, 12, 17], every=2, min_bo=0.2)
    task = ConnectionStatsTask()
    pd.testing.assert_frame_equal(task.run_stream(connectivity_frames(data), request).table,
                                  task.run(data, request).table, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("smooth", [None, "ma", "ema"])
@pytest.mark.parametrize("window", [1, 4, 5])
@pytest.mark.parametrize("min_run", [1, 2, 4])
def test_ordered_bond_events_match_legacy_smoothing_and_flicker(smooth, window, min_run):
    data = connectivity()
    request = BondEventsRequest(threshold=0.5, hysteresis=0.2, smooth=smooth, window=window,
                                min_run=min_run, frames=[0, 2, 4, 5, 6, 7, 8, 9, 12, 17, 18])
    task = BondEventsTask()
    pd.testing.assert_frame_equal(task.run_stream(connectivity_frames(data), request).table,
                                  task.run(data, request).table, atol=1e-12, rtol=1e-12)


def test_flicker_leading_short_runs_and_long_trace_keep_bounded_state():
    rng = np.random.default_rng(12)
    for _ in range(60):
        values = rng.choice([0.1, 0.9], size=60)
        request = BondEventsRequest(threshold=0.5, hysteresis=0.0, min_run=4, smooth=None)
        trace = BondTraceState(request)
        events = [event for i, value in enumerate(values) for event in trace.add(i, i, value)] + trace.finish()
        states = clean_flicker(schmitt_hysteresis(values, 0.5, 0), 4)
        changes = np.flatnonzero(states[1:] != states[:-1]) + 1
        assert [event[0] for event in events] == list(changes)
        assert len(trace.samples) <= request.window


def test_late_reference_excludes_dependency_and_preserves_requested_stride():
    data = trajectory()
    request = TrajectoryDisplacementSeriesRequest(frames=[0, 2, 5, 7], every=2,
                                                   reference_frame=10, atom_ids=[11, 5], dims=("z", "xyz"))
    task = TrajectoryDisplacementSeriesTask()
    expected = task.run(data, request).table
    actual = task.run_stream(frame_stream(data), request, pipeline=pipeline(task, request)).table
    pd.testing.assert_frame_equal(actual, expected)
    assert set(actual.frame_index) == {0, 5}


def test_peak_rss_does_not_scale_with_trajectory_length():
    from reaxkit.core.runtime.benchmark import isolated_case
    kwargs = dict(payload_bytes=1024 * 1024, workers=2, max_in_flight=4)
    short = isolated_case(frames=12, **kwargs)
    long = isolated_case(frames=160, **kwargs)
    assert long.peak_rss_bytes <= short.peak_rss_bytes + 32 * 1024 * 1024


def test_disabled_lazy_detail_is_never_constructed(tmp_path):
    from reaxkit.core.runtime.artifacts import TableChunks
    def forbidden():
        raise AssertionError("Disabled detail producer was consumed")
        yield
    with ArtifactWriter(tmp_path, [ArtifactSpec("detail", "detail.parquet", "detail", False, "parquet")]) as writer:
        writer.write_chunks("detail", TableChunks(forbidden()))


def test_cross_command_artifact_overwrite_is_rejected(tmp_path):
    spec = ArtifactSpec("core", "result.csv", "core", True, "csv")
    with ArtifactWriter(tmp_path, [spec], metadata={"command": "first"}) as writer:
        writer.append("core", [{"value": 1}])
    with pytest.raises(FileExistsError, match="belongs to command"):
        with ArtifactWriter(tmp_path, [spec], overwrite=True, metadata={"command": "second"}) as writer:
            writer.append("core", [{"value": 2}])
    assert pd.read_csv(tmp_path / "result.csv").value.tolist() == [1]


@pytest.mark.parametrize("group", ["three_folded_wurtzite", "four_folded_wurtzite"])
def test_neighbor_stream_matches_serial_periodic_geometry(group):
    from importlib import import_module
    module = import_module(f"reaxkit.analysis.ferroelectrics.{group}.neighbors")
    xyz = np.array([[0, 0, 0], [1.7, 0, -0.5], [-0.85, 1.472, -0.5], [-0.85, -1.472, -0.5], [0, 0, 1.8]])
    data = TrajectoryData(positions=np.stack([xyz, xyz + 0.1, xyz + 0.2]),
                          atom_ids=[2, 4, 6, 8, 10], elements=["Al", "N", "N", "N", "N"],
                          iterations=np.array([0, 10, 20]),
                          simulation=SimulationData(atom_ids=[2, 4, 6, 8, 10], iterations=np.array([0, 10, 20]),
                                                    cell_lengths=np.full((3, 3), 10.), cell_angles=np.full((3, 3), 90.)))
    request = module.WurtziteNeighborRequest(frames=[0, 2], charge_source="formal", formal_charges={"Al": 3, "N": -3})
    task = module.WurtziteNeighborTask()
    expected = task.run(data, request)
    actual = task.run_stream(frame_stream(data), request, pipeline=pipeline(task, request))
    pd.testing.assert_frame_equal(actual.centers, expected.centers)
    pd.testing.assert_frame_equal(actual.neighbors, expected.neighbors)


@pytest.mark.parametrize("kind", ["coordination", "hybridization", "connection_list"])
@pytest.mark.parametrize("workers", [1, 3])
def test_connectivity_frame_maps_preserve_categories_and_atom_ids(kind, workers):
    from reaxkit.analysis.connectivity.coordination import CoordinationStatusTask, CoordinationStatusRequest
    from reaxkit.analysis.connectivity.hybridization import HybridizationStatusTask, HybridizationStatusRequest
    from reaxkit.analysis.connectivity.connectivity import ConnectionListTask, ConnectionListRequest
    from reaxkit.domain.data_models import CoordinationStatusBundleData
    data = connectivity()
    selection = dict(frames=[1, 4, 8, 12, 16], every=2)
    if kind == "coordination":
        task, request = CoordinationStatusTask(), CoordinationStatusRequest(valences={"C": 4, "N": 3, "H": 1}, **selection)
        complete = CoordinationStatusBundleData(connectivity=data, force_field_parameters=None)
        source = (CoordinationStatusBundleData(connectivity=frame, force_field_parameters=None) for frame in connectivity_frames(data))
    else:
        complete, source = data, connectivity_frames(data)
        task, request = (HybridizationStatusTask(), HybridizationStatusRequest(hybridizations={"sp": 1, "sp2": 2, "sp3": 3}, **selection)) if kind == "hybridization" else (ConnectionListTask(), ConnectionListRequest(**selection))
    actual = task.run_stream(source, request, pipeline=pipeline(task, request, workers)).table
    pd.testing.assert_frame_equal(actual, task.run(complete, request).table)


@pytest.mark.parametrize("geometry", [False, True])
def test_scipy_voronoi_parallel_geometry_matches_serial(geometry):
    from reaxkit.analysis.trajectory.voronoi import VoronoiRequest, VoronoiScipyTask, VoronoiGeometryScipyTask
    data = trajectory()
    data.positions = np.random.default_rng(42).random((13, 24, 3)) * 10
    data.atom_ids, data.elements = list(range(10, 34)), ["C"] * 24
    data.simulation.atom_ids = data.atom_ids
    request = VoronoiRequest(frames=[0, 3, 8], atom_ids=[12, 16, 18])
    task = VoronoiGeometryScipyTask() if geometry else VoronoiScipyTask()
    actual = task.run_stream(frame_stream(data), request, pipeline=pipeline(task, request)).table
    pd.testing.assert_frame_equal(actual, task.run(data, request).table)


def test_charge_series_parallel_preserves_nonfinite_values():
    from reaxkit.analysis.timeseries.timeseries import ChargeSeriesTask, ChargeSeriesRequest
    from reaxkit.domain.data_models import ChargeData
    charges = np.array([[0.1, np.nan], [0.2, 0.3], [0.5, -0.1]])
    simulation = SimulationData(atom_ids=[1, 2], iterations=np.arange(3), elements=["C", "H"])
    data = ChargeData(charges=charges, iterations=np.arange(3), simulation=simulation)
    source = (replace(data, charges=charges[i:i+1], iterations=np.array([i]),
                      simulation=replace(simulation, iterations=np.array([i])),
                      metadata={"source_frame_indices": [i]}) for i in range(3))
    task, request = ChargeSeriesTask(), ChargeSeriesRequest()
    pd.testing.assert_frame_equal(task.run_stream(source, request, pipeline=pipeline(task, request)).table,
                                  task.run(data, request).table)


def test_json_and_extxyz_are_published_only_at_commit(tmp_path):
    specs = [ArtifactSpec("json", "metadata.json", "core", True, "json"),
             ArtifactSpec("xyz", "trajectory.xyz", "core", True, "extxyz")]
    with ArtifactWriter(tmp_path, specs) as writer:
        writer.write_json("json", {"frames": [2, 7]})
        writer.write_file("xyz", lambda path: path.write_text("1\nframe=2\nH 0 0 0\n", encoding="utf-8"))
        assert not (tmp_path / "trajectory.xyz").exists()
    assert json.loads((tmp_path / "metadata.json").read_text()) == {"frames": [2, 7]}
    assert (tmp_path / "trajectory.xyz").read_text().startswith("1\n")


def test_affinity_selection_explicit_serial_and_unsupported_processes(monkeypatch):
    import reaxkit.core.runtime.execution_contracts as contracts
    monkeypatch.setattr(contracts.os, "sched_getaffinity", lambda _: {2, 4, 6}, raising=False)
    task = SimpleNamespace(execution_capabilities=TaskCapabilities(shape=ExecutionShape.INDEPENDENT_FRAME_MAP, thread_safe=True))
    request = SimpleNamespace(frames=[3, 8], every=1)
    assert resolve_execution_policy(task, request, environ={}).workers == 2
    serial = resolve_execution_policy(task, request, {"execution": "serial"}, environ={})
    assert serial.workers == 1 and serial.decision_reason == "explicit_serial_execution"
    processes = resolve_execution_policy(task, request, {"execution": "processes"}, environ={})
    assert processes.workers == 1 and "process" in processes.decision_reason


def test_native_threads_are_scoped_and_restored():
    task = TrajectoryCoordinateSeriesTask()
    before = [(info["prefix"], info["num_threads"]) for info in threadpool_info()]
    results = list(pipeline(task, SimpleNamespace()).map_ordered(range(4), lambda _: threadpool_info()))
    assert all(info["num_threads"] == 1 for result in results for info in result.value)
    assert [(info["prefix"], info["num_threads"]) for info in threadpool_info()] == before


def test_automatic_workers_require_benchmark_gate_but_allow_expert_override():
    task, request = TrajectoryCoordinateSeriesTask(), TrajectoryCoordinateSeriesRequest()
    env = {"SLURM_CPUS_PER_TASK": "4"}
    automatic = resolve_execution_policy(task, request, environ=env)
    assert automatic.workers == 1 and automatic.worker_source == "benchmark_serial"
    assert resolve_execution_policy(task, request, {"workers": 3}, environ=env).workers == 3
    assert resolve_execution_policy(task, request, {"execution": "threads"}, environ=env).workers == 4


@pytest.mark.parametrize("order", [[0, 2, 4, 7], [4, 0, 2, 7]])
def test_reference_first_reader_replays_public_source_order(order):
    from reaxkit.core.runtime.reference_frames import reference_frames
    source = ({"source": value} for value in order)
    with reference_frames(source, 4, index_fn=lambda data, _: data["source"]) as (reference, selected):
        assert reference["source"] == 4
        assert [index for index, _ in selected] == [0, 2, 4, 7]


def test_nested_cli_policy_preserves_parent_overrides():
    import argparse
    from reaxkit.core.runtime.cli_policy import add_execution_arguments
    parser = argparse.ArgumentParser()
    parser.add_subparsers(dest="action").add_parser("run")
    add_execution_arguments(parser)
    args = parser.parse_args(["--workers", "3", "--output-profile", "full", "run"])
    assert args.workers == 3 and args.output_profile == "full"
    assert parser.parse_args(["run", "--workers", "auto"]).workers == 0


def test_empty_parquet_preserves_typed_schema(tmp_path):
    empty = pd.DataFrame({"frame_index": pd.Series(dtype="int64"), "value": pd.Series(dtype="float64")})
    with ArtifactWriter(tmp_path, [ArtifactSpec("core", "core.parquet", "core", True, "parquet")]) as writer:
        writer.write_table("core", empty)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "core.parquet"), empty)


def test_artifact_commit_failure_rolls_back_and_cleans_temporary_files(tmp_path, monkeypatch):
    import reaxkit.core.runtime.artifacts as artifacts
    specs = [ArtifactSpec(name, f"{name}.csv", "core", True, "csv") for name in ("a", "b")]
    (tmp_path / "a.csv").write_text("previous\n")
    original = artifacts.os.replace
    def fail_second(source, destination):
        if destination == tmp_path / "b.csv":
            raise OSError("disk unavailable")
        return original(source, destination)
    monkeypatch.setattr(artifacts.os, "replace", fail_second)
    with pytest.raises(OSError, match="disk unavailable"):
        with ArtifactWriter(tmp_path, specs, overwrite=True) as writer:
            writer.append("a", [{"v": 1}])
            writer.append("b", [{"v": 2}])
    assert (tmp_path / "a.csv").read_text() == "previous\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["a.csv"]


def test_legacy_detail_format_empty_schema_and_no_overwrite(tmp_path):
    spec = ArtifactSpec("detail", "detail.parquet", "detail", False, "parquet", units={"distance": "angstrom"})
    with ArtifactWriter(tmp_path, [spec], profile="legacy") as writer:
        writer.write_table("detail", pd.DataFrame(columns=["distance"]))
    assert (tmp_path / "detail.csv").read_text().strip() == "distance"
    manifest = json.loads((tmp_path / "reaxkit_artifacts.json").read_text())
    assert manifest["artifacts"][0]["units"] == {"distance": "angstrom"}
    with pytest.raises(FileExistsError):
        with ArtifactWriter(tmp_path, [spec], profile="legacy") as writer:
            writer.append("detail", [{"distance": 1.0}])
    assert not list(tmp_path.glob(".rk-*"))
