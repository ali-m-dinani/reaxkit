from __future__ import annotations

import csv

import numpy as np

from reaxkit.analysis.ferroelectrics.dynamic_charge import (
    DETAIL_COLUMNS,
    SUMMARY_COLUMNS,
    DynamicChargeChangeRequest,
    DynamicChargeChangeTask,
    calculate_dynamic_charge_changes,
)
from reaxkit.domain.data_models import ChargeData, SimulationData


def _charge_data() -> ChargeData:
    simulation = SimulationData(
        atom_ids=[10, 20],
        elements=["O", "Ti"],
        iterations=np.array([0, 5, 10]),
        time=np.array([0.0, 0.25, 0.5]),
    )
    return ChargeData(
        charges=np.array([[-1.0, 2.0], [-0.8, 1.9], [-1.2, 2.3]]),
        iterations=np.array([0, 5, 10]),
        simulation=simulation,
    )


def test_charge_changes_use_frame_zero_and_build_both_tables() -> None:
    result = calculate_dynamic_charge_changes(_charge_data(), DynamicChargeChangeRequest())

    assert list(result.charges.columns) == DETAIL_COLUMNS
    assert list(result.summary.columns) == SUMMARY_COLUMNS
    oxygen = result.charges[result.charges["atom_number"] == 10]
    assert np.allclose(oxygen["delta_charge"], [0.0, 0.2, -0.2])
    oxygen_summary = result.summary[result.summary["atom_number"] == 10].iloc[0]
    assert np.isclose(oxygen_summary["mean(charge)"], -1.0)
    assert np.isclose(oxygen_summary["range(delta_charge)"], 0.4)
    frame_zero = result.summary_per_frame_for_all_atoms.iloc[0]
    assert frame_zero["frame"] == 0
    assert np.isclose(frame_zero["mean(charge)"], 0.5)
    assert np.isclose(frame_zero["min(charge)"], -1.0)
    assert np.isclose(frame_zero["max(charge)"], 2.0)
    assert result.summary_per_frame_per_atom_type.shape[0] == 6


def test_filtered_frames_still_compare_with_unfiltered_frame_zero() -> None:
    request = DynamicChargeChangeRequest(atom_numbers=[20], selected_frames=[1, 2])
    result = calculate_dynamic_charge_changes(_charge_data(), request)

    assert result.charges["frame"].tolist() == [1, 2]
    assert np.allclose(result.charges["delta_charge"], [-0.1, 0.3])
    assert result.iterations.tolist() == [5, 10]
    assert np.allclose(result.time_values, [0.25, 0.5])


def test_streamed_charge_changes_match_materialized_result() -> None:
    data = _charge_data()
    request = DynamicChargeChangeRequest(atom_numbers=[10], selected_frames=[1, 2])

    def frames():
        for frame in range(3):
            simulation = SimulationData(
                atom_ids=[10, 20],
                elements=["O", "Ti"],
                iterations=np.asarray([data.iterations[frame]]),
                time=np.asarray([data.simulation.time[frame]]),
            )
            yield ChargeData(
                charges=data.charges[frame: frame + 1],
                iterations=np.asarray([data.iterations[frame]]),
                simulation=simulation,
                metadata={"source_frame_indices": [frame]},
            )

    expected = calculate_dynamic_charge_changes(data, request)
    actual = DynamicChargeChangeTask().run_stream(frames(), request)

    assert actual.charges.equals(expected.charges)
    assert actual.summary.equals(expected.summary)
    assert actual.summary_per_frame_for_all_atoms.equals(
        expected.summary_per_frame_for_all_atoms
    )
    assert actual.summary_per_frame_per_atom_type.equals(
        expected.summary_per_frame_per_atom_type
    )
    assert actual.frame_indices.tolist() == expected.frame_indices.tolist()
    assert actual.iterations.tolist() == expected.iterations.tolist()
    assert np.allclose(actual.time_values, expected.time_values)


def test_streamed_analysis_includes_atoms_added_after_frame_zero() -> None:
    def frames():
        yield ChargeData(
            charges=np.asarray([[0.1, -0.1]]),
            iterations=np.asarray([0]),
            simulation=SimulationData(atom_ids=[1, 2], elements=["H", "O"]),
            metadata={"source_frame_indices": [0]},
        )
        yield ChargeData(
            charges=np.asarray([[0.2, -0.2, 0.3]]),
            iterations=np.asarray([10]),
            simulation=SimulationData(atom_ids=[1, 2, 3], elements=["H", "O", "C"]),
            metadata={"source_frame_indices": [1]},
        )

    result = DynamicChargeChangeTask().run_stream(frames(), DynamicChargeChangeRequest())

    added = result.charges[result.charges["atom_number"] == 3]
    assert added["frame"].tolist() == [1]
    assert added["atom_type"].tolist() == ["C"]
    assert np.isnan(added["delta_charge"].iloc[0])
    assert result.summary["atom_number"].tolist() == [1, 2, 3]


def test_streaming_chunks_detail_and_retains_only_a_small_preview(tmp_path) -> None:
    detail_path = tmp_path / "charges.csv"
    matrix_path = tmp_path / "charges.dat"
    progress_updates = []

    def frames():
        for frame in range(12):
            yield ChargeData(
                charges=np.asarray([[frame / 10, -frame / 10]]),
                iterations=np.asarray([frame * 5]),
                simulation=SimulationData(atom_ids=[10, 20], elements=["O", "Ti"]),
                metadata={"source_frame_indices": [frame]},
            )

    request = DynamicChargeChangeRequest(
        _detail_csv_path=str(detail_path),
        _matrix_path=str(matrix_path),
        _expected_frames=12,
        _retain_matrix=True,
    )
    result = DynamicChargeChangeTask().run_stream(
        frames(),
        request,
        reporter=lambda *update: progress_updates.append(update),
    )

    try:
        with detail_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert len(rows) == 25  # header plus 12 frames x 2 atoms
        assert result.detail_row_count == 24
        assert len(result.charges) == 20
        assert result.matrix_columns.tolist() == [0, 1]
        assert matrix_path.is_file()
        assert any(current == 12 and total == 12 for _, current, total, _ in progress_updates)
        oxygen = result.summary[result.summary["atom_number"] == 10].iloc[0]
        assert np.isclose(oxygen["mean(charge)"], 0.55)
        assert np.isclose(oxygen["median(charge)"], 0.55)
    finally:
        matrix_path.unlink(missing_ok=True)


def test_streaming_can_skip_the_detailed_csv(tmp_path) -> None:
    matrix_path = tmp_path / "charges.dat"

    def frames():
        for frame in range(2):
            yield ChargeData(
                charges=np.asarray([[1.0 + frame, 3.0 + frame, 5.0 + frame]]),
                iterations=np.asarray([frame]),
                simulation=SimulationData(
                    atom_ids=[1, 2, 3],
                    elements=["Al", "N", "Al"],
                ),
                metadata={"source_frame_indices": [frame]},
            )

    result = DynamicChargeChangeTask().run_stream(
        frames(),
        DynamicChargeChangeRequest(
            _detail_csv_path=None,
            _matrix_path=str(matrix_path),
        ),
    )

    assert result.detail_csv_path is None
    assert len(result.charges) == 6
    assert not (tmp_path / "charges.csv").exists()
    frame_zero = result.summary_per_frame_for_all_atoms.iloc[0]
    assert np.isclose(frame_zero["mean(charge)"], 3.0)
    al_frame_zero = result.summary_per_frame_per_atom_type.query(
        "frame == 0 and atom_type == 'Al'"
    ).iloc[0]
    assert np.isclose(al_frame_zero["mean(charge)"], 3.0)
    assert np.isclose(al_frame_zero["min(charge)"], 1.0)
    assert np.isclose(al_frame_zero["max(charge)"], 5.0)
