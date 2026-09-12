from __future__ import annotations

import numpy as np

from reaxkit.analysis.ferroelectrics.charge_field import (
    TABLE_COLUMNS,
    ChargeFieldRequest,
    ChargeFieldTask,
    calculate_charge_field_response,
)
from reaxkit.core.platform.constants import const
from reaxkit.domain.data_models import ChargeData, ElectricFieldData, SimulationData


def _field_data() -> ElectricFieldData:
    # Deliberately use a different order than the charge frames. Alignment must
    # be by iteration, never by row position.
    return ElectricFieldData(
        applied_field_values=np.asarray([[0.3], [0.1], [0.2]]),
        applied_field_components=("field_z",),
        sampled_field_iterations=np.asarray([20, 0, 10]),
    )


def test_charge_field_response_matches_electric_field_by_iteration() -> None:
    charges = ChargeData(
        charges=np.asarray([[1.0, -1.0], [1.1, -1.1], [1.2, -1.2]]),
        iterations=np.asarray([0, 10, 20]),
        simulation=SimulationData(atom_ids=[7, 9], elements=["Ti", "O"]),
    )
    result = calculate_charge_field_response(
        charges,
        _field_data(),
        ChargeFieldRequest(atom_numbers=[9], field_direction="z"),
    )

    scale = float(const("electric_field_VA_to_MVcm"))
    assert list(result.table.columns) == TABLE_COLUMNS
    assert result.table["atom_number"].tolist() == [9, 9, 9]
    assert result.table["atom_type"].tolist() == ["O", "O", "O"]
    assert np.allclose(result.table["charge"], [-1.0, -1.1, -1.2])
    assert np.allclose(result.table["delta_charge"], [0.0, -0.1, -0.2])
    assert np.allclose(result.table["electric_field"], np.asarray([0.1, 0.2, 0.3]) * scale)
    assert result.baseline_charges == {9: -1.0}


def test_streamed_charge_field_loads_frame_zero_for_later_frame_delta() -> None:
    def frames():
        yield ChargeData(
            charges=np.asarray([[0.1]]),
            iterations=np.asarray([0]),
            simulation=SimulationData(atom_ids=[7], elements=["Ti"]),
            metadata={"source_frame_indices": [0]},
        )
        yield ChargeData(
            charges=np.asarray([[0.4]]),
            iterations=np.asarray([20]),
            simulation=SimulationData(atom_ids=[7], elements=["Ti"]),
            metadata={"source_frame_indices": [2]},
        )

    result = ChargeFieldTask(_field_data()).run_stream(
        frames(),
        ChargeFieldRequest(atom_numbers=[7], frames=[2]),
    )

    assert result.table["frame"].tolist() == [2]
    assert result.table["charge"].tolist() == [0.4]
    assert np.allclose(result.table["delta_charge"], [0.3])
    assert result.baseline_charges == {7: 0.1}


def test_streamed_charge_field_supports_atom_added_in_a_later_frame() -> None:
    def frames():
        yield ChargeData(
            charges=np.asarray([[0.1]]),
            iterations=np.asarray([0]),
            simulation=SimulationData(atom_ids=[1], elements=["H"]),
            metadata={"source_frame_indices": [0]},
        )
        yield ChargeData(
            charges=np.asarray([[0.2, -0.3]]),
            iterations=np.asarray([10]),
            simulation=SimulationData(atom_ids=[1, 5], elements=["H", "O"]),
            metadata={"source_frame_indices": [1]},
        )

    field = ElectricFieldData(
        applied_field_values=np.asarray([[0.0], [0.4]]),
        applied_field_components=("field_z",),
        sampled_field_iterations=np.asarray([0, 10]),
    )
    result = ChargeFieldTask(field).run_stream(
        frames(),
        ChargeFieldRequest(atom_numbers=[5]),
    )

    assert result.table["frame"].tolist() == [1]
    assert result.table["atom_number"].tolist() == [5]
    assert result.table["atom_type"].tolist() == ["O"]
    assert result.table["charge"].tolist() == [-0.3]
    assert np.isnan(result.table["delta_charge"].iloc[0])
    assert result.baseline_charges == {}
