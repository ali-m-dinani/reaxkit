from __future__ import annotations

import numpy as np
import pytest

from reaxkit.analysis.ferroelectrics.binned_dynamic_charge import (
    AVERAGE_COLUMNS,
    HEATMAP_CMAP,
    BinnedDynamicChargeRequest,
    BinnedDynamicChargeTask,
    calculate_binned_dynamic_charges,
    generate_binned_charge_heatmaps,
)
from reaxkit.domain.data_models import ChargeData, ElectrostaticsData, SimulationData, TrajectoryData


def _data() -> ElectrostaticsData:
    simulation = SimulationData(
        atom_ids=[10, 20, 30], iterations=np.asarray([0, 5]),
        time=np.asarray([0.0, 0.25]), elements=["O", "Ti", "O"],
    )
    trajectory = TrajectoryData(
        positions=np.asarray([
            [[0.0, -5.0, 0.0], [2.0, 0.0, 2.0], [0.0, 8.0, 0.0]],
            [[2.0, 100.0, 2.0], [0.0, -100.0, 0.0], [2.0, 0.0, 2.0]],
        ]),
        elements=["O", "Ti", "O"], atom_ids=[10, 20, 30],
        iterations=np.asarray([0, 5]), simulation=simulation,
    )
    charges = ChargeData(
        charges=np.asarray([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]),
        iterations=np.asarray([0, 5]), simulation=simulation,
    )
    return ElectrostaticsData(trajectory=trajectory, charges=charges)


def test_xz_bins_are_fixed_from_frame_zero_and_sum_over_y() -> None:
    result = calculate_binned_dynamic_charges(
        _data(), BinnedDynamicChargeRequest(plane="xz", bins_x=2, bins_z=2)
    )

    assert result.plane == "xz"
    assert result.axis_labels == ("x", "z")
    assert result.atom_bin_numbers == {10: 1, 20: 4, 30: 1}
    frame_one = result.table[result.table["frame"] == 1]
    assert frame_one["charge"].tolist() == [10.0 + 30.0, 0.0, 0.0, 20.0]
    assert frame_one["delta_charge"].tolist() == [36.0, 0.0, 0.0, 18.0]
    assert frame_one["atom_count"].tolist() == [2, 0, 0, 1]
    assert frame_one["bin_y"].tolist() == [-1, -1, -1, -1]
    assert frame_one["y_center"].isna().all()
    assert np.allclose(result.time_values, [0.0, 0.25])
    assert all(column not in result.table for column in AVERAGE_COLUMNS)


def test_average_flag_adds_per_particle_average_columns() -> None:
    result = calculate_binned_dynamic_charges(
        _data(),
        BinnedDynamicChargeRequest(
            plane="xz", bins_x=2, bins_z=2, average=True
        ),
    )

    frame_one = result.table[result.table["frame"] == 1]
    assert frame_one["average_charge"].tolist()[0] == 20.0
    assert frame_one["average_delta_charge"].tolist()[0] == 18.0
    assert np.isnan(frame_one["average_charge"].tolist()[1])
    assert HEATMAP_CMAP == "coolwarm_r"


def test_selected_plane_requires_only_its_two_bin_counts() -> None:
    with pytest.raises(ValueError, match="xz plane requires bins_z"):
        calculate_binned_dynamic_charges(
            _data(), BinnedDynamicChargeRequest(plane="xz", bins_x=2)
        )


def test_heatmaps_are_written_for_both_quantities_and_all_frames(tmp_path) -> None:
    result = calculate_binned_dynamic_charges(
        _data(), BinnedDynamicChargeRequest(plane="xz", bins_x=2, bins_z=2)
    )
    written = generate_binned_charge_heatmaps(result, tmp_path, dpi=60, progress=False)

    assert len(written) == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)
    assert {path.parent.name for path in written} == {"charge", "delta_charge"}


def test_average_heatmaps_replace_sum_heatmaps(tmp_path) -> None:
    result = calculate_binned_dynamic_charges(
        _data(),
        BinnedDynamicChargeRequest(
            plane="xz", bins_x=2, bins_z=2, average=True
        ),
    )
    written = generate_binned_charge_heatmaps(
        result, tmp_path, dpi=60, progress=False
    )

    assert {path.parent.name for path in written} == {
        "average_charge",
        "average_delta_charge",
    }


def test_streamed_quick_charge_frames_match_materialized_result() -> None:
    data = _data()
    request = BinnedDynamicChargeRequest(
        plane="xz", bins_x=2, bins_z=2, average=True, _expected_frames=2
    )

    def frames():
        for frame in range(2):
            trajectory = TrajectoryData(
                positions=data.trajectory.positions[frame: frame + 1],
                elements=data.trajectory.elements,
                atom_ids=data.trajectory.atom_ids,
                iterations=np.asarray([data.trajectory.iterations[frame]]),
                source_frame_indices=np.asarray([frame]),
                simulation=SimulationData(
                    atom_ids=data.trajectory.atom_ids,
                    elements=data.trajectory.elements,
                    iterations=np.asarray([data.trajectory.iterations[frame]]),
                ),
            )
            charges = ChargeData(
                charges=data.charges.charges[frame: frame + 1],
                iterations=np.asarray([data.charges.iterations[frame]]),
                simulation=trajectory.simulation,
                metadata={"charges_only": True, "source_frame_indices": [frame]},
            )
            yield ElectrostaticsData(trajectory=trajectory, charges=charges)

    updates = []
    actual = BinnedDynamicChargeTask().run_stream(
        frames(), request, reporter=lambda *update: updates.append(update)
    )
    expected = calculate_binned_dynamic_charges(data, request)

    assert actual.table.equals(expected.table)
    assert actual.atom_bin_numbers == expected.atom_bin_numbers
    assert any(current == 2 and total == 2 for _, current, total, _ in updates)
