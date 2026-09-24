from __future__ import annotations

import numpy as np

from reaxkit.analysis.ferroelectrics.charge_extxyz import (
    ChargeExtendedXYZRequest,
    ChargeExtendedXYZTask,
)
from reaxkit.core.platform.constants import const
from reaxkit.domain.data_models import (
    ChargeData,
    ElectricFieldData,
    ElectrostaticsData,
    SimulationData,
    TrajectoryData,
)


def _frame(
        source_frame: int,
        iteration: int,
        elements: list[str],
        positions: list[list[float]],
        charges: list[float],
) -> ElectrostaticsData:
    atom_ids = list(range(1, len(elements) + 1))
    simulation = SimulationData(
        atom_ids=atom_ids,
        elements=elements,
        iterations=np.asarray([iteration]),
        cell_lengths=np.asarray([[5.0, 5.0, 8.0]]),
        cell_angles=np.asarray([[90.0, 90.0, 90.0]]),
    )
    trajectory = TrajectoryData(
        positions=np.asarray([positions], dtype=float),
        elements=elements,
        atom_ids=atom_ids,
        simulation=simulation,
        iterations=np.asarray([iteration]),
        source_frame_indices=np.asarray([source_frame]),
    )
    charge_data = ChargeData(
        charges=np.asarray([charges], dtype=float),
        simulation=simulation,
        iterations=np.asarray([iteration]),
        metadata={"source_frame_indices": [source_frame]},
    )
    return ElectrostaticsData(trajectory=trajectory, charges=charge_data)


def test_streamed_charge_extxyz_uses_frame_zero_baseline_and_later_atoms(tmp_path) -> None:
    output = tmp_path / "charges.extxyz"
    frames = [
        _frame(0, 0, ["Al", "N"], [[0, 0, 0], [1, 0, 0]], [0.1, -0.1]),
        _frame(
            1,
            10,
            ["Al", "N", "H"],
            [[0, 0, 0.1], [1, 0, 0.1], [2, 0, 0.1]],
            [0.25, -0.2, 0.05],
        ),
    ]
    result = ChargeExtendedXYZTask().run_stream(
        iter(frames),
        ChargeExtendedXYZRequest(_output_path=str(output)),
    )
    lines = output.read_text(encoding="utf-8").splitlines()

    assert result.frame_indices.tolist() == [0, 1]
    assert result.iterations.tolist() == [0, 10]
    assert result.table["frames_written"].iloc[0] == 2
    assert lines[0] == "2"
    assert lines[4] == "3"
    frame_one_atoms = [line.split() for line in lines[6:9]]
    assert np.isclose(float(frame_one_atoms[0][-1]), 0.15)
    assert np.isclose(float(frame_one_atoms[1][-1]), -0.1)
    assert frame_one_atoms[2][-1] == "nan"


def test_selected_frames_still_load_frame_zero_as_reference(tmp_path) -> None:
    output = tmp_path / "selected.extxyz"
    data = [
        _frame(0, 0, ["Al"], [[0, 0, 0]], [0.1]),
        _frame(2, 20, ["Al"], [[0, 0, 0.2]], [0.4]),
    ]
    result = ChargeExtendedXYZTask().run_stream(
        iter(data),
        ChargeExtendedXYZRequest(frames=[2], _output_path=str(output)),
    )

    assert result.frame_indices.tolist() == [2]
    atom_line = output.read_text(encoding="utf-8").splitlines()[2].split()
    assert np.isclose(float(atom_line[-1]), 0.3)


def test_streamed_charge_extxyz_writes_iteration_matched_field_metadata(tmp_path) -> None:
    output = tmp_path / "field.extxyz"
    electric_field = ElectricFieldData(
        applied_field_values=np.asarray([[0.3], [0.1]]),
        applied_field_components=("field_z",),
        sampled_field_iterations=np.asarray([10, 0]),
    )
    frames = [
        _frame(0, 0, ["Al"], [[0, 0, 0]], [0.1]),
        _frame(1, 10, ["Al"], [[0, 0, 0.1]], [0.2]),
    ]
    for frame in frames:
        frame.electric_field = electric_field

    ChargeExtendedXYZTask().run_stream(
        iter(frames),
        ChargeExtendedXYZRequest(
            include_electric_field=True,
            field_direction="z",
            _output_path=str(output),
        ),
    )

    headers = output.read_text(encoding="utf-8").splitlines()[1::3]
    scale = float(const("electric_field_VA_to_MVcm"))
    assert f"electric_field={0.1 * scale}" in headers[0]
    assert f"electric_field={0.3 * scale}" in headers[1]
    assert "electric_field_direction=z" in headers[0]
    assert "electric_field_units=MV/cm" in headers[0]
