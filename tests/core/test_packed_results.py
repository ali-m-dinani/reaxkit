from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from reaxkit.core.runtime.packed_results import (
    PackedResultStore, publish_packed_results, remove_packed_checkpoint,
)
from reaxkit.core.runtime.result_store import ResultStore
from reaxkit.presentation.workflow_artifacts import register_checkpoint, workflow_artifact_policy


def populate(store):
    for position in range(20):
        store.append(position, position, {"table": pd.DataFrame({"frame_index": [position], "value": [position / 3]})})
    store.transition("analysis_complete")


@pytest.mark.parametrize("format_name", ["csv", "parquet"])
def test_success_compacts_and_preserves_lazy_tables(tmp_path, format_name):
    directory = tmp_path / "recovery-test" / "checkpoint"
    cache = tmp_path / "cache.results.zip"
    with workflow_artifact_policy(Namespace()):
        store = ResultStore(directory, {}, batch_frames=1, format_name=format_name)
        register_checkpoint(store, cache_destination=cache)
        populate(store)
        table = store.table("table")
        expected = table.materialize()
        assert len(list(directory.rglob("*"))) > 40
        assert not cache.exists()
    assert cache.is_file()
    assert not directory.parent.exists()
    assert list(tmp_path.iterdir()) == [cache]
    pd.testing.assert_frame_equal(table.materialize(), expected)
    with PackedResultStore(cache, {}) as packed:
        pd.testing.assert_frame_equal(packed.table("table").materialize(), expected)
        assert packed.table("table").select_frame(5).frame_index.tolist() == [5]
        with pytest.raises(RuntimeError, match="closed"):
            packed.append(20, 20, {})


@pytest.mark.parametrize("failure", ["export", "handled", "interrupt", "packing", "verification"])
def test_failures_preserve_recovery(tmp_path, monkeypatch, failure):
    directory = tmp_path / "checkpoint"
    cache = tmp_path / "cache.results.zip"
    args = Namespace()

    def fail(*unused_args, **unused_kwargs):
        raise OSError("injected failure")

    if failure == "packing":
        monkeypatch.setattr(ZipFile, "write", fail)
    elif failure == "verification":
        monkeypatch.setattr(PackedResultStore, "validate", fail)

    def command():
        with workflow_artifact_policy(args):
            store = ResultStore(directory, {}, batch_frames=1)
            register_checkpoint(store, cache_destination=cache)
            populate(store)
            if failure == "export":
                fail()
            elif failure == "handled":
                args._checkpoint_exit_code = 1
            elif failure == "interrupt":
                raise KeyboardInterrupt()

    if failure == "handled":
        command()
    else:
        with pytest.raises(KeyboardInterrupt if failure == "interrupt" else OSError):
            command()
    assert not cache.exists()
    assert not list(tmp_path.glob("*.tmp"))
    with ResultStore(directory, {}, resume=True) as store:
        assert store.manifest["state"] == "analysis_complete"
        assert len(store.table("table")) == 20


def test_disabled_cache_retains_checkpoint(tmp_path):
    with workflow_artifact_policy(Namespace()):
        store = ResultStore(tmp_path / "checkpoint", {}, batch_frames=1)
        register_checkpoint(store)
        populate(store)
    with ResultStore(tmp_path / "checkpoint", {}, resume=True) as store:
        assert store.manifest["state"] == "complete"
        assert len(store.table("table")) == 20


def test_cleanup_does_not_remove_unrelated_files(tmp_path):
    directory = tmp_path / "checkpoint"
    with workflow_artifact_policy(Namespace()):
        store = ResultStore(directory, {}, batch_frames=1)
        register_checkpoint(store, cache_destination=tmp_path / "cache.results.zip")
        populate(store)
        (directory / "keep.txt").write_text("user data")
    assert (directory / "keep.txt").read_text() == "user data"
    assert len(store.table("table")) == 20


def test_cleanup_failure_leaves_verified_cache_and_live_tables(tmp_path, monkeypatch):
    from reaxkit.core.runtime import packed_results

    def fail(directory):
        raise OSError("injected deletion failure")

    monkeypatch.setattr(packed_results.shutil, "rmtree", fail)
    cache = tmp_path / "cache.results.zip"
    with workflow_artifact_policy(Namespace()):
        store = ResultStore(tmp_path / "checkpoint", {}, batch_frames=1)
        register_checkpoint(store, cache_destination=cache)
        populate(store)
        table = store.table("table")
    assert len(table.materialize()) == 20
    with PackedResultStore(cache, {}) as packed:
        assert packed.committed_frames == 20


def test_cache_identity_tracks_inputs_and_settings_but_not_output(tmp_path):
    from reaxkit.core.runtime.checkpoint_policy import checkpoint_identity
    from reaxkit.analysis.ferroelectrics.charge_extxyz import ChargeExtendedXYZTask, ChargeExtendedXYZRequest

    source = tmp_path / "input"
    source.write_text("first")
    output = tmp_path / "output.extxyz"
    task = ChargeExtendedXYZTask()
    request = ChargeExtendedXYZRequest(_output_path=str(output))
    args = {"output_profile": "standard"}
    identity_source = {"sources": [{"path": str(source)}], "adapter": "ReaxFFAdapter"}
    identity = checkpoint_identity(task, request, args, identity_source)
    output.write_text("saved trajectory")
    assert checkpoint_identity(task, request, args, identity_source) == identity
    assert checkpoint_identity(task, replace(request, precision=request.precision + 1), args, identity_source) != identity
    assert checkpoint_identity(task, request, {"output_profile": "full"}, identity_source) != identity
    source.write_text("other")
    assert checkpoint_identity(task, request, args, identity_source) != identity


def test_archive_corruption_is_detected(tmp_path):
    cache = tmp_path / "cache.results.zip"
    with ResultStore(tmp_path / "checkpoint", {}, batch_frames=1) as store:
        populate(store)
        publish_packed_results(store, cache).close()
    with ZipFile(cache, "a") as archive:
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("batches/batch-00000000/table-000.parquet", b"damaged")
    with pytest.raises(ValueError, match="Damaged"):
        PackedResultStore(cache, {})


@pytest.mark.parametrize("kind", ["count", "mean", "max", "events", "dihedral", "charge", "potential", "local"])
def test_task_replay_without_original_checkpoint_or_input(tmp_path, kind):
    from tests.core.test_unified_rollout import connectivity, connectivity_frames, trajectory, frame_stream
    from tests.analysis.test_charge_extxyz import _frame
    from tests.core.test_checkpoint_commands import add_field
    from reaxkit.analysis.connectivity.connectivity import (
        ConnectionStatsTask, ConnectionStatsRequest, BondEventsTask, BondEventsRequest,
    )
    from reaxkit.analysis.trajectory.dihedral import DihedralTask, DihedralRequest
    from reaxkit.analysis.ferroelectrics.charge_extxyz import ChargeExtendedXYZTask, ChargeExtendedXYZRequest
    from reaxkit.analysis.electrostatics.potential_and_electric_field.trajectory import (
        PotentialElectricFieldTrajectoryTask, PotentialElectricFieldTrajectoryRequest,
    )
    if kind in {"count", "mean", "max", "events"}:
        task, request = ((BondEventsTask(), BondEventsRequest(min_run=1)) if kind == "events" else
                         (ConnectionStatsTask(), ConnectionStatsRequest(how=kind)))
        frames = list(connectivity_frames(connectivity()))
    elif kind == "dihedral":
        task, request = DihedralTask(), DihedralRequest(atom_ids=[2, 5, 8, 11])
        frames = list(frame_stream(trajectory()))
    elif kind == "local":
        from tests.analysis.test_hbn_reference_polarization import _trajectory_from_reference
        from reaxkit.analysis.ferroelectrics.hbn_reference.polarization import REFERENCE_STRUCTURE_PATH
        from reaxkit.analysis.ferroelectrics.hbn_reference.local_polarization import (
            HBNReferenceLocalPolarizationTask, HBNReferenceLocalPolarizationRequest,
        )
        task = HBNReferenceLocalPolarizationTask()
        request = HBNReferenceLocalPolarizationRequest(reference_path=REFERENCE_STRUCTURE_PATH,
            replication=(2, 1, 1), charge_source="formal", volume_method="cell")
        request._write_extxyz = True
        frames = [_trajectory_from_reference(REFERENCE_STRUCTURE_PATH)]
        frames[0].source_frame_indices = [0]
    else:
        output = str(tmp_path / "original.extxyz")
        task, request = ((ChargeExtendedXYZTask(), ChargeExtendedXYZRequest(_output_path=output)) if kind == "charge" else
            (PotentialElectricFieldTrajectoryTask(), PotentialElectricFieldTrajectoryRequest(_output_path=output,
                gamma_by_symbol={"al": 1.2, "n": 1.5}, periodic=(False, False, False))))
        frames = [_frame(index, index * 10, ["Al", "N"], [[0, 0, index * .1], [2, 0, 1]], [.1, -.1]) for index in range(3)]
        add_field(frames)
    directory = tmp_path / "checkpoint"
    store = ResultStore(directory, {}, batch_frames=1)
    request._result_store = store
    result = task.run_stream(iter(frames), request)
    expected = result.table.materialize() if hasattr(result.table, "materialize") else result.table.copy()
    cache = tmp_path / "cache.results.zip"
    packed = publish_packed_results(store, cache)
    store.transition("complete")
    remove_packed_checkpoint(store, packed)
    assert not directory.exists()
    request = replace(request)
    request._result_store = packed
    if kind in {"charge", "potential"}:
        original = Path(request._output_path).read_bytes()
        Path(request._output_path).unlink()
        request._output_path = str(tmp_path / "replayed.extxyz")
    actual = task.run_stream(iter(()), request)
    table = actual.table.materialize() if hasattr(actual.table, "materialize") else actual.table
    if kind in {"charge", "potential"}:
        assert Path(request._output_path).read_bytes() == original
        assert table.frames_written.tolist() == expected.frames_written.tolist()
    else:
        pd.testing.assert_frame_equal(table, expected)
    if kind == "local":
        from reaxkit.analysis.ferroelectrics.hbn_reference.local_polarization import write_local_polarization_extxyz
        write_local_polarization_extxyz(result, tmp_path / "expected.extxyz")
        write_local_polarization_extxyz(actual, tmp_path / "actual.extxyz")
        assert (tmp_path / "actual.extxyz").read_bytes() == (tmp_path / "expected.extxyz").read_bytes()
    packed.close()


def test_executor_reuses_cache_and_honors_resume(tmp_path, monkeypatch):
    from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
    from reaxkit.analysis.timeseries.timeseries import ChargeSeriesTask, ChargeSeriesRequest
    from reaxkit.engine.reaxff.adapter import ReaxFFAdapter
    from tests.core.test_streaming_analysis import FIXTURE_DIR

    args = {"run_dir": str(FIXTURE_DIR), "fort7": str(FIXTURE_DIR / "fort.7"),
            "engine": "reaxff", "cache_dir": str(tmp_path / "cache"),
            "project_root": str(tmp_path / "workspace"), "output_profile": "standard", "progress": False}
    task = ChargeSeriesTask()
    with workflow_artifact_policy(Namespace()):
        result = AnalysisExecutor().run(task, ChargeSeriesRequest(frames=[0, 2]), dict(args))
        expected = result.table.materialize()
    assert len(list((tmp_path / "cache").rglob("*.results.zip"))) == 1
    assert not list((tmp_path / "workspace").rglob("checkpoint"))

    def unexpected_stream(*unused_args, **unused_kwargs):
        raise AssertionError("Input frames must not be read on a cache hit")

    monkeypatch.setattr(ReaxFFAdapter, "stream", unexpected_stream)
    with workflow_artifact_policy(Namespace()):
        actual = AnalysisExecutor().run(task, ChargeSeriesRequest(frames=[0, 2]), dict(args))
        pd.testing.assert_frame_equal(actual.table.materialize(), expected)
    with pytest.raises(Exception, match="manifest.json"):
        AnalysisExecutor().run(task, ChargeSeriesRequest(frames=[0, 2]),
                               {**args, "resume": str(tmp_path / "missing-checkpoint")})


def test_successful_resume_packs_checkpoint_after_failed_export(tmp_path, monkeypatch):
    from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
    from reaxkit.analysis.timeseries.timeseries import ChargeSeriesTask, ChargeSeriesRequest
    from tests.core.test_streaming_analysis import FIXTURE_DIR

    args = {"run_dir": str(FIXTURE_DIR), "fort7": str(FIXTURE_DIR / "fort.7"),
            "engine": "reaxff", "cache_dir": str(tmp_path / "cache"),
            "project_root": str(tmp_path / "workspace"), "output_profile": "standard", "progress": False}
    task = ChargeSeriesTask()
    with pytest.raises(OSError, match="export failed"):
        with workflow_artifact_policy(Namespace()):
            result = AnalysisExecutor().run(task, ChargeSeriesRequest(frames=[0, 2]), dict(args))
            expected = result.table.materialize()
            raise OSError("export failed")
    checkpoint = next((tmp_path / "workspace").rglob("checkpoint"))
    assert not list((tmp_path / "cache").rglob("*.results.zip"))

    def unexpected_calculation(*unused_args, **unused_kwargs):
        raise AssertionError("Committed calculations must not run again")

    monkeypatch.setattr(task, "run", unexpected_calculation)
    with workflow_artifact_policy(Namespace()):
        actual = AnalysisExecutor().run(task, ChargeSeriesRequest(frames=[0, 2]), {**args, "resume": str(checkpoint)})
        pd.testing.assert_frame_equal(actual.table.materialize(), expected)
    assert not checkpoint.exists()
    assert len(list((tmp_path / "cache").rglob("*.results.zip"))) == 1
