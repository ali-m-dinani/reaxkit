"""Acceptance gates for the remaining execution/output migrations."""

from dataclasses import replace
from types import SimpleNamespace
import io
import json
import numpy as np
import pandas as pd
import pytest

from reaxkit.domain.data_models import SimulationData, TrajectoryData
from reaxkit.analysis.stress_strain.z_binned_deformation_gradient_strain import ZBinnedDeformationGradientStrainTask, ZBinnedDeformationGradientStrainRequest
from reaxkit.analysis.stress_strain.z_binned_strain_using_top_bottom_atoms import ZBinnedTopBottomStrainTask, ZBinnedTopBottomStrainRequest
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.presentation.workflow_artifacts import write_workflow_csv, workflow_artifact_policy


def strain_data():
    rng = np.random.default_rng(170)
    base = rng.uniform(0.5, 7.5, (32, 3))
    xyz = np.stack([(base * (1 + frame * .002) + [frame * .7, 0, 0]) % 10 for frame in range(7)])
    sim = SimulationData(atom_ids=list(range(3, 35)), iterations=np.arange(7) * 20,
                         cell_lengths=np.full((7, 3), 10.), cell_angles=np.full((7, 3), 90.))
    return TrajectoryData(positions=xyz, atom_ids=sim.atom_ids, elements=["Al"]*32, iterations=sim.iterations, simulation=sim)


def single_frames(data):
    for frame in range(len(data.positions)):
        yield replace(data, positions=data.positions[frame:frame+1], iterations=data.iterations[frame:frame+1],
                      source_frame_indices=np.array([frame]),
                      simulation=replace(data.simulation, iterations=data.iterations[frame:frame+1],
                                         cell_lengths=data.simulation.cell_lengths[frame:frame+1],
                                         cell_angles=data.simulation.cell_angles[frame:frame+1]))


@pytest.mark.parametrize("unwrap", [False, True])
@pytest.mark.parametrize("bin_range", ["reference", "current"])
@pytest.mark.parametrize("kind", ["gradient", "span"])
def test_strain_stream_preserves_selected_frames_and_periodic_history(unwrap, bin_range, kind):
    data = strain_data()
    task_type, request_type = ((ZBinnedDeformationGradientStrainTask, ZBinnedDeformationGradientStrainRequest)
                              if kind == "gradient" else (ZBinnedTopBottomStrainTask, ZBinnedTopBottomStrainRequest))
    task = task_type()
    request = request_type(z_bins=2, selected_frames=[1, 3, 6], every=2, unwrap=unwrap, bin_range=bin_range)
    pd.testing.assert_frame_equal(task.run_stream(single_frames(data), request).table,
                                  task.run(data, request).table, atol=1e-12, rtol=1e-12)
    dependencies = AnalysisExecutor._requested_frame_indices(request, TrajectoryData, task)
    assert dependencies == (list(range(7)) if unwrap else [0, 1, 3, 6])


@pytest.mark.parametrize("named", [False, True])
def test_workflow_csv_adapter_preserves_public_index_bytes(tmp_path, named):
    frame = pd.DataFrame({"value": [1.5, 2.5]}, index=pd.Index([7, 11], name="atom" if named else None))
    expected = frame.to_csv(index=True)
    path = tmp_path / "table.csv"
    with workflow_artifact_policy(SimpleNamespace(command="test", output_profile="standard")):
        write_workflow_csv(frame, path, index=True)
    assert path.read_bytes() == expected.encode()
    manifest = json.loads((tmp_path / "table.csv.artifacts.json").read_text())
    assert manifest["run_metadata"]["command"] == "test"


def test_workflow_context_restores_profile_after_failure(tmp_path):
    path = tmp_path / "summary.csv"
    with pytest.raises(RuntimeError):
        with workflow_artifact_policy(SimpleNamespace(command="test", output_profile="minimal")):
            assert write_workflow_csv(pd.DataFrame({"x": [1]}), path, tier="summary") is None
            raise RuntimeError("stop")
    write_workflow_csv(pd.DataFrame({"x": [1]}), path, tier="summary")
    assert path.is_file()


@pytest.mark.parametrize("threshold", [0, 1, 3])
@pytest.mark.parametrize("selection", [None, [1, 3, 4, 7]])
def test_molecular_lifetime_stream_matches_serial_segments(threshold, selection):
    from reaxkit.analysis.molecular_analysis.molecular_analysis import MoleculeLifetimeTask, MoleculeLifetimeRequest
    from reaxkit.domain.data_models import MolecularAnalysisData
    rows = [{"iter": frame*10, "molecular_formula": formula, "freq": frequency, "molecular_mass": 18.}
            for frame in range(9) for formula, frequency in (("H2O", [0, 2, 3, 0, 4, 4, 0, 1, 0][frame]), ("OH", max(0, frame-3))) if frequency]
    complete = MolecularAnalysisData(iterations=np.arange(9)*10, totals=pd.DataFrame(), molecular_species=pd.DataFrame(rows))
    def source():
        for index, iteration in enumerate(complete.iterations):
            data = MolecularAnalysisData(iterations=np.array([iteration]), totals=pd.DataFrame(),
                    molecular_species=complete.molecular_species.query("iter == @iteration").reset_index(drop=True))
            data.source_frame_indices = np.array([index])
            yield data
    task, request = MoleculeLifetimeTask(), MoleculeLifetimeRequest(frames=selection, every=2, min_freq=threshold)
    pd.testing.assert_frame_equal(task.run_stream(source(), request).table, task.run(complete, request).table)


def test_molecular_parser_stream_matches_materialized_normalization(tmp_path):
    from reaxkit.engine.reaxff.adapter_parts.molecular_stream import iter_molecular_data
    from reaxkit.engine.reaxff.adapter_parts.normalizers import _molecular_analysis_from_molfra_handler
    from reaxkit.engine.reaxff.io.molfra_handler import MolFraHandler
    path = tmp_path / "molfra.out"
    path.write_text("\n".join(f"Iteration {frame}\n{frame} 2 x H2O 18.0\n{frame} 1 x OH 17.0\nTotal number of molecules 3\nTotal number of atoms 8\nTotal system molecular mass 53.0" for frame in (0, 10, 20)))
    expected = _molecular_analysis_from_molfra_handler(MolFraHandler(path))
    actual = list(iter_molecular_data(path))
    pd.testing.assert_frame_equal(pd.concat([item.molecular_species for item in actual], ignore_index=True), expected.molecular_species)
    pd.testing.assert_frame_equal(pd.concat([item.totals for item in actual], ignore_index=True)[expected.totals.columns], expected.totals)


@pytest.mark.parametrize("family", ["three_folded_wurtzite", "four_folded_wurtzite"])
def test_reference_polarity_stream_matches_late_reference(family):
    from importlib import import_module
    module = import_module(f"reaxkit.analysis.ferroelectrics.{family}.polarity")
    points = np.array([[0, 0, 0], [1.7, 0, -.5], [-.85, 1.472, -.5], [-.85, -1.472, -.5], [0, 0, 1.8]])
    sim = SimulationData(atom_ids=[2, 4, 6, 8, 10], iterations=np.arange(4)*10,
                         cell_lengths=np.full((4, 3), 10.), cell_angles=np.full((4, 3), 90.))
    data = TrajectoryData(positions=np.stack([points+i*.1 for i in range(4)]), atom_ids=sim.atom_ids,
                          elements=["Al", "N", "N", "N", "N"], iterations=sim.iterations, simulation=sim)
    request = module.WurtzitePolarityRequest(frames=[0, 2], reference_frame=3, charge_source="formal", formal_charges={"Al": 3, "N": -3})
    task = module.WurtzitePolarityTask()
    actual, expected = task.run_stream(single_frames(data), request), task.run(data, request)
    for name in ("table", "centers", "neighbors", "neighbor_geometry", "summary", "proton_summary"):
        pd.testing.assert_frame_equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("kind", ["basal_dipole", "basal_binned", "basal_local", "three_binned"])
def test_basal_polarization_variants_stream_match_serial(kind):
    from reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.dipole import BasalPlaneDipoleTask, BasalPlaneDipoleRequest
    from reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.polarization import BasalPlanePolarizationTask, BasalPlanePolarizationRequest
    from reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.local_polarization import BasalPlaneLocalPolarizationTask, BasalPlaneLocalPolarizationRequest
    from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.polarization import BinnedPolarizationTask, BinnedPolarizationRequest
    types = {"basal_dipole": (BasalPlaneDipoleTask, BasalPlaneDipoleRequest), "basal_binned": (BasalPlanePolarizationTask, BasalPlanePolarizationRequest),
             "basal_local": (BasalPlaneLocalPolarizationTask, BasalPlaneLocalPolarizationRequest), "three_binned": (BinnedPolarizationTask, BinnedPolarizationRequest)}
    points = np.array([[0, 0, 0], [1.7, 0, -.5], [-.85, 1.472, -.5], [-.85, -1.472, -.5], [0, 0, 1.8]])
    sim = SimulationData(atom_ids=[2, 4, 6, 8, 10], iterations=np.arange(4)*10,
                         cell_lengths=np.full((4, 3), 10.), cell_angles=np.full((4, 3), 90.))
    data = TrajectoryData(positions=np.stack([points+i*.1 for i in range(4)]), atom_ids=sim.atom_ids,
                          elements=["Al", "N", "N", "N", "N"], iterations=sim.iterations, simulation=sim)
    task_type, request_type = types[kind]
    task, request = task_type(), request_type(frames=[1, 3], reference_frame=2, charge_source="formal", formal_charges={"Al": 3, "N": -3})
    actual, expected = task.run_stream(single_frames(data), request), task.run(data, request)
    for name in ("table", "summary"):
        pd.testing.assert_frame_equal(getattr(actual, name), getattr(expected, name), atol=1e-12, rtol=1e-12)
    if hasattr(actual, "trajectory"):
        np.testing.assert_allclose(actual.trajectory.positions[[1, 3]], data.positions[[1, 3]])
        actual._trajectory_spool.close()


@pytest.mark.parametrize("unwrap", [False, True])
@pytest.mark.parametrize("origin", ["first", 4])
def test_blocked_diffusivity_matches_per_atom_fit(unwrap, origin):
    from reaxkit.analysis.trajectory.diffusivity import DiffusivityTask, DiffusivityRequest
    data = strain_data()
    data.positions[2, 1, 0] = np.nan  # Unselected padded atoms must not affect the fit.
    request = DiffusivityRequest(frames=[0, 2, 4, 6], origin=origin, atom_ids=[3, 7, 9], unwrap=unwrap)
    task = DiffusivityTask()
    pd.testing.assert_frame_equal(task.run_blocks(single_frames(data), request).table,
                                  task.run(data, request).table, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ["dominant", "mass", "composition", "frequency", "totals"])
@pytest.mark.parametrize("edge_cases", [False, True])
def test_molecular_frame_maps_match_materialized(kind, edge_cases):
    from reaxkit.analysis.molecular_analysis import molecular_analysis as mol
    from reaxkit.analysis.timeseries import timeseries as series
    from reaxkit.domain.data_models import MolecularAnalysisData
    types = {"dominant": (mol.DominantSpeciesTask, mol.DominantSpeciesRequest()),
             "mass": (mol.LargestMoleculeByMassTask, mol.LargestMoleculeByMassRequest()),
             "composition": (mol.LargestMoleculeCompositionTask, mol.LargestMoleculeCompositionRequest()),
             "frequency": (series.MolecularFrequencySeriesTask, series.MolecularFrequencySeriesRequest(molecules=["H2O", "OH"])),
             "totals": (series.MolecularTotalsSeriesTask, series.MolecularTotalsSeriesRequest())}
    rows = [{"iter": frame*10, "molecular_formula": formula, "freq": frame+1, "molecular_mass": mass}
            for frame in range(5) for formula, mass in (("H2O", 18.), ("OH", 17.))]
    totals = pd.DataFrame({"iter": np.arange(5)*10, "total_molecules": 2*(np.arange(5)+1),
                           "total_atoms": 5*(np.arange(5)+1), "total_molecular_mass": 35*(np.arange(5)+1)})
    data = MolecularAnalysisData(iterations=np.arange(5)*10, totals=totals, molecular_species=pd.DataFrame(rows))
    if edge_cases:
        data.molecular_species.loc[2, "freq"] = np.nan
        data.molecular_species.loc[3, "molecular_mass"] = np.nan
        data.molecular_species.loc[8:9, ["freq", "molecular_mass"]] = [4., 18.]
    def source():
        for index, iteration in enumerate(data.iterations):
            frame = MolecularAnalysisData(iterations=np.array([iteration]), totals=totals.iloc[index:index+1].reset_index(drop=True),
                                          molecular_species=data.molecular_species.query("iter == @iteration").reset_index(drop=True))
            frame.source_frame_indices = np.array([index])
            yield frame
    task_type, request = types[kind]
    request = replace(request, frames=[1, 2, 4], every=2)
    expected, actual = task_type().run(data, request).table, task_type().run_stream(source(), request).table
    columns = [column for column in ("frame_index", "molecules", "quantity", "element") if column in expected]
    pd.testing.assert_frame_equal(actual.sort_values(columns).reset_index(drop=True), expected.sort_values(columns).reset_index(drop=True))


@pytest.mark.parametrize("family", ["three_folded_wurtzite", "four_folded_wurtzite"])
@pytest.mark.parametrize("profile", ["standard", "minimal", "full", "legacy"])
def test_polarity_detail_profiles_stream_and_preserve_primary(tmp_path, family, profile):
    from importlib import import_module
    module = import_module(f"reaxkit.analysis.ferroelectrics.{family}.polarity")
    writer = import_module(f"reaxkit.workflows.ferroelectrics.{family}.artifacts")
    points = np.array([[0, 0, 0], [1.7, 0, -.5], [-.85, 1.472, -.5], [-.85, -1.472, -.5], [0, 0, 1.8]])
    sim = SimulationData(atom_ids=[2, 4, 6, 8, 10], iterations=np.arange(4)*10,
                         cell_lengths=np.full((4, 3), 10.), cell_angles=np.full((4, 3), 90.))
    data = TrajectoryData(positions=np.stack([points+i*.1 for i in range(4)]), atom_ids=sim.atom_ids,
                          elements=["Al", "N", "N", "N", "N"], iterations=sim.iterations, simulation=sim)
    request = module.WurtzitePolarityRequest(frames=[0, 2], reference_frame=3, charge_source="formal", formal_charges={"Al": 3, "N": -3})
    expected = module.WurtzitePolarityTask().run(data, request)
    request._output_profile = profile
    result = module.WurtzitePolarityTask().run_stream(single_frames(data), request)
    assert result.neighbor_geometry.empty
    pd.testing.assert_frame_equal(result.table, expected.table)
    paths = writer.write_polarity_tables(result, tmp_path, args=SimpleNamespace(output_profile=profile))
    if profile in {"full", "legacy"}:
        table = pd.read_parquet(paths["neighbors"]) if profile == "full" else pd.read_csv(paths["neighbors"])
        assert len(table) == len(expected.neighbor_geometry)
        assert set(table.frame_index) == {0, 2}
        assert paths["neighbors"].suffix == (".parquet" if profile == "full" else ".csv")
    else:
        assert not hasattr(result, "detail_chunks")
        assert not paths["neighbors"].exists()
    assert paths["summary"].exists() is (profile != "minimal")


def test_result_trajectory_spool_preserves_time_and_labels():
    from reaxkit.core.runtime.trajectory_spool import TrajectorySpool
    data = strain_data()
    spool = TrajectorySpool()
    try:
        for index, frame in enumerate(single_frames(data)):
            frame.atom_labels = np.array([["Al_special" if index == 3 else "Al"] * len(frame.atom_ids)])
            frame.simulation.time = np.array([index * .025])
            spool.append(index, frame)
        result = spool.finish()
        np.testing.assert_array_equal(result.positions, data.positions)
        np.testing.assert_allclose(result.simulation.time, np.arange(7)*.025)
        assert result.atom_labels[3, 0] == "Al_special"
    finally:
        spool.close()


def test_csv_row_adapter_preserves_legacy_values_and_rolls_back(tmp_path):
    import csv
    from reaxkit.presentation.workflow_artifacts import workflow_csv_rows
    rows = [[1, None, "quoted, value"], [2, 1.25, "line\nbreak"]]
    expected = io.StringIO(newline="")
    writer = csv.writer(expected)
    writer.writerow(["index", "value", "text"])
    writer.writerows(rows)
    path = tmp_path / "rows.csv"
    with workflow_csv_rows(path, ["index", "value", "text"]) as writer:
        for row in rows:
            writer.writerow(row)
    assert list(csv.reader(io.StringIO(path.read_text()))) == list(csv.reader(io.StringIO(expected.getvalue())))
    before = path.read_bytes()
    with pytest.raises(RuntimeError):
        with workflow_csv_rows(path, ["index", "value", "text"]) as writer:
            for _ in range(2049):
                writer.writerow([1, 2, 3])
            raise RuntimeError("cancel")
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".rk-*"))


def test_relabel_stream_preserves_primary_labels_and_source_frames():
    from reaxkit.analysis.trajectory.relabel import TrajectoryRelabelByCoordinationTask, TrajectoryRelabelByCoordinationRequest
    from reaxkit.domain.data_models import ConnectivityData, ConnectivityTrajectoryData
    data = strain_data()
    bonds = np.zeros((7, 32, 32))
    bonds[1::2, :, :] = .05
    connectivity = ConnectivityData(bond_orders=bonds, atom_ids=data.atom_ids,
                                    elements=data.elements, iterations=data.iterations)
    bundle = ConnectivityTrajectoryData(trajectory=data, connectivity=connectivity)
    def source():
        for i, frame in enumerate(single_frames(data)):
            yield replace(bundle, trajectory=frame, connectivity=replace(connectivity,
                    bond_orders=bonds[i:i+1], iterations=data.iterations[i:i+1], source_frame_indices=np.array([i])))
    request = TrajectoryRelabelByCoordinationRequest(frames=[1, 3, 6], valences={"Al": 1}, mode="by_type")
    task = TrajectoryRelabelByCoordinationTask()
    expected, actual = task.run(bundle, request), task.run_stream(source(), request)
    try:
        pd.testing.assert_frame_equal(actual.table, expected.table)
        np.testing.assert_array_equal(actual.trajectory.positions, expected.trajectory.positions)
        np.testing.assert_array_equal(actual.trajectory.atom_labels, expected.trajectory.atom_labels)
        np.testing.assert_array_equal(actual.trajectory.source_frame_indices, [1, 3, 6])
    finally:
        actual._trajectory_spool.close()


def test_scientific_stream_memory_is_bounded_in_fresh_processes():
    from pathlib import Path
    import subprocess
    import sys
    script = Path(__file__).resolve().parents[2] / "benchmarks/rollout_completion.py"
    results = []
    for count in (16, 512):
        process = subprocess.run([sys.executable, str(script), "--memory-child", str(count)],
                                 check=True, capture_output=True, text=True)
        results.append(json.loads(process.stdout))
        assert results[-1]["source_frames_read"] == count
    assert results[1]["peak_rss_bytes"] <= results[0]["peak_rss_bytes"] + 32 * 1024 * 1024


def test_streamed_extxyz_failure_preserves_existing_output(tmp_path):
    from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.trajectory import PolarityExtendedXYZRequest, PolarityExtendedXYZTask
    output = tmp_path / "polarity.extxyz"
    output.write_text("previous completed trajectory\n")
    data = strain_data()
    data.elements = ["Al"] + ["N"] * 31
    def broken():
        yield next(single_frames(data))
        raise RuntimeError("reader failed")
    request = PolarityExtendedXYZRequest(_output_path=str(output), reference_frame=0,
                                        charge_source="formal", formal_charges={"Al": 3, "N": -3})
    with pytest.raises(RuntimeError, match="reader failed"):
        PolarityExtendedXYZTask().run_stream(broken(), request)
    assert output.read_text() == "previous completed trajectory\n"
    assert not list(tmp_path.glob(".rk-*"))


def test_dynamic_charge_failure_aborts_detail_and_scratch(tmp_path):
    from reaxkit.analysis.ferroelectrics.dynamic_charge import DynamicChargeChangeRequest, DynamicChargeChangeTask
    from reaxkit.domain.data_models import ChargeData
    output = tmp_path / "detail.parquet"
    output.write_bytes(b"previous artifact")
    def broken():
        yield ChargeData(charges=np.array([[1.]]), iterations=np.array([0]), metadata={"source_frame_indices": [0]})
        raise RuntimeError("cancelled")
    request = DynamicChargeChangeRequest(_detail_csv_path=str(output), _matrix_path=str(tmp_path / "matrix.bin"))
    with pytest.raises(RuntimeError, match="cancelled"):
        DynamicChargeChangeTask().run_stream(broken(), request)
    assert output.read_bytes() == b"previous artifact"
    assert {path.name for path in tmp_path.iterdir()} == {"detail.parquet"}


@pytest.mark.parametrize("available", [False, True])
def test_connectivity_trajectory_stream_loads_optional_valences_once(available, monkeypatch):
    from pathlib import Path
    from reaxkit.engine.reaxff.adapter import ReaxFFAdapter
    from reaxkit.domain.data_models import ConnectivityTrajectoryData
    from reaxkit.analysis.trajectory.relabel import _empty_force_field_parameters
    fixture = Path(__file__).resolve().parents[1] / "fixtures/reaxff_isomer_representatives_detection"
    adapter = ReaxFFAdapter()
    parameters, calls = _empty_force_field_parameters(), []
    def load(args, reporter=None):
        calls.append(args["ffield"])
        if not available:
            raise FileNotFoundError("optional force field")
        return parameters
    monkeypatch.setattr(adapter, "load_force_field", load)
    frames = list(adapter.stream(ConnectivityTrajectoryData, {
        "run_dir": str(fixture), "_frame_indices": [0, 1], "input_cache": False,
    }))
    assert len(frames) == 2 and len(calls) == 1
    assert all(frame.force_field_parameters is (parameters if available else None) for frame in frames)


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("profile", [None, "standard", "full", "legacy"])
@pytest.mark.parametrize("charge_source", ["formal", "reaxff"])
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_hbn_polarization_stream_prepares_reference_once(local, profile, charge_source, workers, monkeypatch, tmp_path):
    from ase.io import read
    from reaxkit.domain.data_models import ChargeData, ElectrostaticsData
    from reaxkit.core.runtime.frame_pipeline import BoundedFramePipeline
    from reaxkit.core.runtime.execution_contracts import resolve_execution_policy
    from reaxkit.engine.common.generators.structure_transformers import orthogonalize_hexagonal_cell
    from reaxkit.analysis.ferroelectrics.hbn_reference import polarization as module
    from reaxkit.analysis.ferroelectrics.hbn_reference.local_polarization import HBNReferenceLocalPolarizationTask, HBNReferenceLocalPolarizationRequest
    reference = orthogonalize_hexagonal_cell(read(module.REFERENCE_STRUCTURE_PATH)).repeat((2, 1, 1))
    points = reference.positions.copy()
    sim = SimulationData(atom_ids=list(range(1, len(points)+1)), iterations=np.arange(3)*10,
                         cell_lengths=np.tile(reference.cell.lengths(), (3, 1)), cell_angles=np.tile(reference.cell.angles(), (3, 1)))
    xyz = np.stack([points.copy() for _ in range(3)])
    xyz[1:, np.array(reference.get_chemical_symbols()) == "Al", 2] += .02
    data = TrajectoryData(positions=xyz, atom_ids=sim.atom_ids, elements=reference.get_chemical_symbols(), iterations=sim.iterations, simulation=sim)
    task_type, request_type = (HBNReferenceLocalPolarizationTask, HBNReferenceLocalPolarizationRequest) if local else (module.HBNReferencePolarizationTask, module.HBNReferencePolarizationRequest)
    task, request = task_type(), request_type(reference_path=module.REFERENCE_STRUCTURE_PATH, frames=[0, 2], reference_frame=1, replication=(2, 1, 1), charge_source=charge_source)
    if not local and profile in {"full", "legacy"}:
        request.include_displacements = True
    frame_data = single_frames(data)
    analysis_data = data
    if charge_source == "reaxff":
        charges = np.array([1., .97, 1.02])[:, None] * np.where(np.array(data.elements) == "Al", 2.4, -2.4)
        analysis_data = ElectrostaticsData(trajectory=data, charges=ChargeData(
            charges=charges, iterations=sim.iterations, simulation=sim))
        frame_data = (ElectrostaticsData(trajectory=frame, charges=ChargeData(
            charges=charges[index:index+1], iterations=frame.iterations, simulation=frame.simulation))
            for index, frame in enumerate(single_frames(data)))
    expected = task.run(analysis_data, request)
    prepare = module.prepare_hbn_reference
    calls = []
    def tracked(*args, **kwargs):
        calls.append(1)
        return prepare(*args, **kwargs)
    monkeypatch.setattr(module, "prepare_hbn_reference", tracked)
    if profile is not None:
        request._output_profile = profile
    runtime = BoundedFramePipeline(resolve_execution_policy(
        task, request, {"workers": workers, "chunk_size": 4}, environ={"SLURM_CPUS_PER_TASK": "4"}))
    actual = task.run_stream(frame_data, request, pipeline=runtime)
    pd.testing.assert_frame_equal(actual.table, expected.table, atol=1e-10, rtol=1e-10)
    assert len(calls) == 1
    target = actual.reference_result if local else actual
    expected_target = expected.reference_result if local else expected
    pd.testing.assert_frame_equal(target.mapping, expected_target.mapping)
    if profile is not None:
        assert target.displacements.empty
        if profile in {"full", "legacy"}:
            from reaxkit.presentation.workflow_artifacts import write_workflow_tables
            paths = write_workflow_tables({tmp_path / "displacements.csv": target.table_chunks["hbn_reference_displacements"]},
                     args=SimpleNamespace(output_profile=profile), details=("displacements.csv",))
            details = pd.read_csv(paths[0]) if profile == "legacy" else pd.read_parquet(paths[0])
            expected_details = expected.reference_result.displacements if local else expected.displacements
            pd.testing.assert_frame_equal(details, expected_details, check_dtype=False, atol=1e-10, rtol=1e-10)
        else:
            assert not getattr(target, "table_chunks", {})
    if local:
        np.testing.assert_allclose(actual.trajectory.positions[[0, 2]], data.positions[[0, 2]])
        actual._trajectory_spool.close()


@pytest.mark.parametrize("profile,explicit,skip,enabled", [
    ("standard", False, False, False), ("minimal", True, False, False),
    ("standard", True, False, True), ("full", False, False, True),
    ("legacy", False, False, True), ("full", False, True, False),
])
def test_dynamic_charge_incremental_detail_profiles(tmp_path, profile, explicit, skip, enabled):
    from reaxkit.analysis.ferroelectrics.dynamic_charge import DynamicChargeChangeRequest, DynamicChargeChangeTask
    from reaxkit.workflows.ferroelectrics.dynamic_charge_workflow import _detail_enabled, _detail_format, _publish_output
    from reaxkit.domain.data_models import ChargeData
    args = SimpleNamespace(output_profile=profile, write_detailed_charges=explicit, skip_detailed_csv=skip)
    assert _detail_enabled(args) is enabled
    charges = np.array([[-1., 2.], [-.8, 1.9], [-1.2, 2.3]])
    sim = SimulationData(atom_ids=[10, 20], elements=["O", "Ti"], iterations=np.array([0, 5, 10]))
    data = ChargeData(charges=charges, iterations=sim.iterations, simulation=sim)
    request = DynamicChargeChangeRequest(selected_frames=[1, 2],
        _detail_csv_path=str(tmp_path / ("staged." + _detail_format(args))) if enabled else None,
        _matrix_path=str(tmp_path / "matrix.bin"))
    frames = (ChargeData(charges=charges[i:i+1], iterations=sim.iterations[i:i+1],
                        simulation=replace(sim, iterations=sim.iterations[i:i+1]), metadata={"source_frame_indices": [i]}) for i in range(3))
    result = DynamicChargeChangeTask().run_stream(frames, request)
    _publish_output(result, args, tmp_path / "published")
    detail = tmp_path / "published" / ("charges." + _detail_format(args))
    assert detail.is_file() is enabled
    if enabled:
        actual = pd.read_csv(detail) if detail.suffix == ".csv" else pd.read_parquet(detail)
        expected = DynamicChargeChangeTask().run(data, request).charges
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
