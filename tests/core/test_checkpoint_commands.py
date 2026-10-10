from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from reaxkit.core.runtime.result_store import ResultStore, ResultTable


def add_field(data):
    from reaxkit.domain.data_models import ElectricFieldData
    for frame in data:
        frame.electric_field = ElectricFieldData(applied_field_values=np.zeros((3, 3)),
            applied_field_components=("field_x", "field_y", "field_z"), sampled_field_iterations=np.array([0, 10, 20]))


def compare_tables(actual, expected):
    for name, table in expected.csv_tables.items():
        saved = actual.csv_tables[name]
        pd.testing.assert_frame_equal(saved.materialize() if isinstance(saved, ResultTable) else saved, table)
    np.testing.assert_array_equal(actual.frame_indices, expected.frame_indices)


@pytest.mark.parametrize("kind", ["local", "projected", "potential", "binned", "charge_field"])
def test_commands_resume_after_input_failure(tmp_path, kind):
    from tests.analysis.test_charge_extxyz import _frame
    from tests.analysis.test_charge_field import _field_data
    from tests.analysis.test_hbn_reference_polarization import _trajectory_from_reference
    from reaxkit.analysis.ferroelectrics.hbn_reference.polarization import REFERENCE_STRUCTURE_PATH
    from reaxkit.analysis.ferroelectrics.hbn_reference.local_polarization import HBNReferenceLocalPolarizationTask, HBNReferenceLocalPolarizationRequest
    from reaxkit.analysis.ferroelectrics.hbn_reference.projected_polarity import HBNReferenceProjectedPolarityTask, HBNReferenceProjectedPolarityRequest
    from reaxkit.analysis.electrostatics.potential_and_electric_field.analysis import PotentialElectricFieldTask, PotentialElectricFieldRequest
    from reaxkit.analysis.ferroelectrics.binned_dynamic_charge import BinnedDynamicChargeTask, BinnedDynamicChargeRequest
    from reaxkit.analysis.ferroelectrics.charge_field import ChargeFieldTask, ChargeFieldRequest

    data = [_frame(index, index * 10, ["Al", "N"], [[0, 0, index * .1], [2, 0, 1]], [.1 + index * .01, -.1]) for index in range(3)]
    add_field(data)
    if kind in {"local", "projected"}:
        data = [_trajectory_from_reference(REFERENCE_STRUCTURE_PATH) for _ in range(3)]
        for index, trajectory in enumerate(data):
            trajectory.source_frame_indices = [index]
        settings = dict(reference_path=REFERENCE_STRUCTURE_PATH, replication=(2, 1, 1), charge_source="formal", volume_method="cell")
        task, request = ((HBNReferenceLocalPolarizationTask(), HBNReferenceLocalPolarizationRequest(**settings)) if kind == "local"
                         else (HBNReferenceProjectedPolarityTask(), HBNReferenceProjectedPolarityRequest(**settings)))
    elif kind == "potential":
        task, request = PotentialElectricFieldTask(), PotentialElectricFieldRequest(gamma_by_symbol={"al": 1.2, "n": 1.5}, periodic=(False, False, False), potential_reference_mode="fixed-midpoint")
    elif kind == "binned":
        task, request = BinnedDynamicChargeTask(), BinnedDynamicChargeRequest(plane="xz", bins_x=2, bins_z=2)
    else:
        task, request = ChargeFieldTask(_field_data()), ChargeFieldRequest(atom_numbers=[1, 2])
        data = [frame.charges for frame in data]
    expected = task.run_stream(iter(data), request)
    if kind == "local":
        request._write_extxyz = True
    def broken():
        yield data[0]
        yield data[1]
        raise OSError("input failure")
    with pytest.raises(OSError, match="input failure"):
        with ResultStore(tmp_path, {}, batch_frames=1) as store:
            request._result_store = store
            task.run_stream(broken(), request)
    with ResultStore(tmp_path, {}, resume=True) as store:
        store.transition("running")
        request._result_store = store
        actual = task.run_stream(iter(data), request)
        compare_tables(actual, expected)
        if kind == "local":
            from reaxkit.analysis.ferroelectrics.hbn_reference.local_polarization import write_local_polarization_extxyz
            write_local_polarization_extxyz(expected, tmp_path / "expected.extxyz")
            write_local_polarization_extxyz(actual, tmp_path / "actual.extxyz")
            assert (tmp_path / "expected.extxyz").read_bytes() == (tmp_path / "actual.extxyz").read_bytes()
            from reaxkit.workflows.ferroelectrics.hbn_reference.local_polarization_workflow import generate_local_2d_plots, generate_local_3d_plots
            assert len(generate_local_2d_plots(actual, tmp_path, plane="xz", component="z", quantity="polarization", bins=(2, 2), global_scaling=True, dpi=40)) == 3
            assert len(generate_local_3d_plots(actual, tmp_path, component="z", quantity="polarization", global_scaling=True, dpi=40)) == 3


@pytest.mark.parametrize("kind", ["charge", "potential"])
def test_trajectory_commits_complete_frames_and_reexports(tmp_path, kind):
    from tests.analysis.test_charge_extxyz import _frame
    from reaxkit.analysis.ferroelectrics.charge_extxyz import ChargeExtendedXYZTask, ChargeExtendedXYZRequest
    from reaxkit.analysis.electrostatics.potential_and_electric_field.trajectory import PotentialElectricFieldTrajectoryTask, PotentialElectricFieldTrajectoryRequest
    data = [_frame(index, index * 10, ["Al", "N"], [[0, 0, index * .1], [2, 0, 1]], [.1 + index * .01, -.1]) for index in range(3)]
    settings = dict(_output_path=str(tmp_path / "expected.extxyz"))
    add_field(data)
    task, request = ((ChargeExtendedXYZTask(), ChargeExtendedXYZRequest(**settings)) if kind == "charge" else
        (PotentialElectricFieldTrajectoryTask(), PotentialElectricFieldTrajectoryRequest(**settings, gamma_by_symbol={"al": 1.2, "n": 1.5}, periodic=(False, False, False))))
    expected = task.run_stream(iter(data), request)
    output = tmp_path / "actual.extxyz"
    request = replace(request, _output_path=str(output))
    with ResultStore(tmp_path / "checkpoint", {}, batch_frames=1) as store:
        request._result_store = store
        actual = task.run_stream(iter(data), request)
        assert output.read_bytes() == (tmp_path / "expected.extxyz").read_bytes()
        assert actual.table.frames_written.iloc[0] == 3
        assert actual.frame_indices.tolist() == expected.frame_indices.tolist()
    output.unlink()
    with ResultStore(tmp_path / "checkpoint", {}, resume=True) as store:
        request._result_store = store
        task.run_stream(iter(()), request)
        assert output.read_bytes() == (tmp_path / "expected.extxyz").read_bytes()


def test_shared_frame_map_resume(tmp_path):
    from tests.core.test_unified_rollout import trajectory, frame_stream
    from reaxkit.analysis.trajectory.dihedral import DihedralTask, DihedralRequest
    task, request = DihedralTask(), DihedralRequest(atom_ids=[2, 5, 8, 11])
    data = trajectory()
    expected = task.run_stream(frame_stream(data), request)
    with ResultStore(tmp_path, {}, batch_frames=2) as store:
        request._result_store = store
        actual = task.run_stream(frame_stream(data), request)
        pd.testing.assert_frame_equal(actual.table.materialize(), expected.table)


@pytest.mark.parametrize("kind", ["count", "mean", "max", "ma", "ema", None])
def test_reducer_and_ordered_state_resume(tmp_path, kind):
    from tests.core.test_unified_rollout import connectivity, connectivity_frames
    from reaxkit.analysis.connectivity.connectivity import ConnectionStatsTask, ConnectionStatsRequest, BondEventsTask, BondEventsRequest
    task, request = ((ConnectionStatsTask(), ConnectionStatsRequest(how=kind)) if kind in {"count", "mean", "max"} else
                     (BondEventsTask(), BondEventsRequest(smooth=kind, min_run=1, window=4, threshold=.5)))
    data = list(connectivity_frames(connectivity()))
    expected = task.run_stream(iter(data), request)
    def broken():
        yield from data[:10]
        raise OSError("interrupted reader")
    with pytest.raises(OSError, match="interrupted reader"):
        with ResultStore(tmp_path, {}, batch_frames=3) as store:
            request._result_store = store
            task.run_stream(broken(), request)
    with ResultStore(tmp_path, {}, resume=True) as store:
        store.transition("running")
        request._result_store = store
        result = task.run_stream(iter(data), request)
        table = result.table.materialize() if isinstance(result.table, ResultTable) else result.table
        pd.testing.assert_frame_equal(table, expected.table)
    with ResultStore(tmp_path, {}, resume=True) as store:
        request._result_store = store
        actual = task.run_stream(iter(()), request)
        table = actual.table.materialize() if isinstance(actual.table, ResultTable) else actual.table
        pd.testing.assert_frame_equal(table, expected.table)
