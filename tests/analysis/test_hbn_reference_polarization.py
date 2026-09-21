from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from reaxkit.analysis.ferroelectrics.hbn_refernce.polarization import (
    HBNReferencePolarizationRequest,
    REFERENCE_STRUCTURE_PATH,
    calculate_hbn_reference_polarization,
)
from reaxkit.core.platform.constants import const
from reaxkit.domain.data_models import (
    ChargeData,
    ElectrostaticsData,
    SimulationData,
    TrajectoryData,
)
from reaxkit.engine.common.generators.structure_transformers import (
    orthogonalize_hexagonal_cell,
)


def _trajectory_from_reference(
        reference_path, *, shift=(0.0, 0.0, 0.0), vacuum_z=0.0
):
    source = read(reference_path)
    assert isinstance(source, Atoms)
    reference = orthogonalize_hexagonal_cell(source)
    reference.wrap()
    reference = reference.repeat((2, 1, 1))
    cell = reference.cell.array.copy()
    fractional = reference.get_scaled_positions(wrap=False)
    strained_cell = cell.copy()
    strained_cell[0] *= 1.02
    positions = fractional @ strained_cell + np.asarray(shift, dtype=float)
    strained_cell[2, 2] += float(vacuum_z)
    symbols = reference.get_chemical_symbols()
    simulation = SimulationData(
        atom_ids=list(range(1, len(reference) + 1)),
        iterations=np.asarray([20]),
        elements=symbols,
        cell_lengths=np.asarray([np.linalg.norm(strained_cell, axis=1)]),
        cell_angles=np.asarray([[90.0, 90.0, 90.0]]),
    )
    return TrajectoryData(
        positions=np.asarray([positions]),
        elements=symbols,
        atom_ids=list(range(1, len(reference) + 1)),
        iterations=np.asarray([20]),
        simulation=simulation,
    )


def test_cell_strain_and_origin_shift_do_not_create_polarization(reference_path) -> None:
    trajectory = _trajectory_from_reference(reference_path, shift=(1.7, -0.8, 0.35))
    request = HBNReferencePolarizationRequest(
        reference_path=reference_path, replication=(2, 1, 1)
    )

    result = calculate_hbn_reference_polarization(trajectory, request)

    assert result.reference.orthogonalized
    assert result.reference.repeats == (2, 1, 1)
    assert result.table.iloc[0]["P_c (uC/cm^2)"] == pytest.approx(0.0, abs=1.0e-9)
    assert np.max(np.abs(result.displacements["displacement_c (angstrom)"])) < 1.0e-10


def test_vacuum_is_not_applied_as_reference_strain(reference_path) -> None:
    trajectory = _trajectory_from_reference(
        reference_path, shift=(1.7, -0.8, 12.0), vacuum_z=25.0
    )
    result = calculate_hbn_reference_polarization(
        trajectory,
        HBNReferencePolarizationRequest(
            reference_path=reference_path,
            replication=(2, 1, 1),
            volume_method="cell",
        ),
    )

    assert result.reference.strain_ratios[2] > 1.15
    assert result.reference.applied_strain_ratios[2] == pytest.approx(1.0)
    assert result.table.iloc[0]["P_c (uC/cm^2)"] == pytest.approx(0.0, abs=1.0e-9)
    assert np.max(np.abs(result.displacements["displacement_c (angstrom)"])) < 1.0e-10


def test_relative_cation_displacement_produces_formal_charge_dipole(reference_path) -> None:
    trajectory = _trajectory_from_reference(reference_path)
    positions = trajectory.positions.copy()
    labels = np.asarray(trajectory.elements)
    positions[0, labels == "Al", 2] += 0.1
    trajectory = TrajectoryData(
        positions=positions,
        elements=trajectory.elements,
        atom_ids=trajectory.atom_ids,
        iterations=trajectory.iterations,
        simulation=trajectory.simulation,
    )
    result = calculate_hbn_reference_polarization(
        trajectory,
        HBNReferencePolarizationRequest(
            reference_path=reference_path,
            replication=(2, 1, 1),
            volume_method="cell",
        ),
    )

    # Removing rigid translation gives +0.05 A on Al and -0.05 A on N.
    expected_dipole = -8 * 3.0 * 0.1
    volume = abs(np.linalg.det(result.reference.atoms.cell.array))
    assert result.table.iloc[0]["dipole_c (e*angstrom)"] == pytest.approx(expected_dipole)
    factor = const("ea3_to_uC_cm2")
    assert factor is not None
    assert result.table.iloc[0]["P_c (uC/cm^2)"] == pytest.approx(
        expected_dipole / volume * factor
    )
    for axis in "xyz":
        assert f"dipole_{axis} (e*angstrom)" in result.table.columns
        assert f"dipole_{axis} (debye)" in result.table.columns
        assert f"P_{axis} (uC/cm^2)" in result.table.columns
        assert f"dipole_{axis} (e*angstrom)" in result.displacements.columns
        assert f"dipole_{axis} (debye)" in result.displacements.columns
    assert result.table.iloc[0]["dipole_z (e*angstrom)"] == pytest.approx(
        expected_dipole
    )
    assert result.table.iloc[0]["P_z (uC/cm^2)"] == pytest.approx(
        expected_dipole / volume * factor
    )
    assert result.table.iloc[0]["electron_charge_sign"] == -1.0


def test_reports_cartesian_and_selected_c_axis_components(reference_path) -> None:
    trajectory = _trajectory_from_reference(reference_path)
    positions = trajectory.positions.copy()
    labels = np.asarray(trajectory.elements)
    shift = np.asarray([0.04, 0.06, 0.08])
    positions[0, labels == "Al"] += shift
    trajectory = TrajectoryData(
        positions=positions,
        elements=trajectory.elements,
        atom_ids=trajectory.atom_ids,
        iterations=trajectory.iterations,
        simulation=trajectory.simulation,
    )
    c_axis = np.asarray([1.0, 1.0, 1.0])
    result = calculate_hbn_reference_polarization(
        trajectory,
        HBNReferencePolarizationRequest(
            reference_path=reference_path,
            replication=(2, 1, 1),
            c_axis=c_axis,
            volume_method="cell",
        ),
    )

    expected = -8 * 3.0 * shift
    row = result.table.iloc[0]
    for component, axis in enumerate("xyz"):
        assert row[f"dipole_{axis} (e*angstrom)"] == pytest.approx(
            expected[component]
        )
    assert row["dipole_c (e*angstrom)"] == pytest.approx(
        expected @ (c_axis / np.linalg.norm(c_axis))
    )


def test_slab_polarization_supports_hull_bbox_and_cell_volumes(reference_path) -> None:
    trajectory = _trajectory_from_reference(reference_path, vacuum_z=25.0)
    positions = trajectory.positions.copy()
    labels = np.asarray(trajectory.elements)
    positions[0, labels == "Al", 2] += 0.1
    trajectory = TrajectoryData(
        positions=positions,
        elements=trajectory.elements,
        atom_ids=trajectory.atom_ids,
        iterations=trajectory.iterations,
        simulation=trajectory.simulation,
    )
    expected_dipole = -8 * 3.0 * 0.1
    factor = const("ea3_to_uC_cm2")
    assert factor is not None
    rows = {}
    for method in ("hull", "bbox", "cell"):
        result = calculate_hbn_reference_polarization(
            trajectory,
            HBNReferencePolarizationRequest(
                reference_path=reference_path,
                replication=(2, 1, 1),
                volume_method=method,
            ),
        )
        row = result.table.iloc[0]
        rows[method] = row
        assert row["volume_method"] == method
        assert np.isfinite(row["volume (angstrom^3)"])
        assert row["volume (angstrom^3)"] > 0.0
        assert row["P_c (uC/cm^2)"] == pytest.approx(
            expected_dipole / row["volume (angstrom^3)"] * factor
        )

    assert rows["hull"]["volume (angstrom^3)"] <= rows["bbox"]["volume (angstrom^3)"]
    assert rows["bbox"]["volume (angstrom^3)"] < rows["cell"]["volume (angstrom^3)"]
    assert rows["cell"]["volume (angstrom^3)"] == pytest.approx(
        rows["cell"]["simulation_cell_volume (angstrom^3)"]
    )


def test_substitution_requires_its_own_formal_charge(reference_path) -> None:
    trajectory = _trajectory_from_reference(reference_path)
    elements = list(trajectory.elements)
    elements[elements.index("Al")] = "B"
    trajectory = TrajectoryData(
        positions=trajectory.positions,
        elements=elements,
        atom_ids=trajectory.atom_ids,
        iterations=trajectory.iterations,
        simulation=trajectory.simulation,
    )

    with pytest.raises(ValueError, match="Missing formal charge.*B"):
        calculate_hbn_reference_polarization(
            trajectory,
            HBNReferencePolarizationRequest(
                reference_path=reference_path, replication=(2, 1, 1)
            ),
        )


def test_reaxff_charge_source_uses_per_frame_charges(reference_path) -> None:
    trajectory = _trajectory_from_reference(reference_path)
    positions = trajectory.positions.copy()
    labels = np.asarray(trajectory.elements)
    positions[0, labels == "Al", 2] += 0.1
    trajectory = TrajectoryData(
        positions=positions,
        elements=trajectory.elements,
        atom_ids=trajectory.atom_ids,
        iterations=trajectory.iterations,
        simulation=trajectory.simulation,
    )
    charges = np.where(labels == "Al", 4.0, -4.0)[None, :]
    data = ElectrostaticsData(
        trajectory=trajectory,
        charges=ChargeData(
            charges=charges,
            iterations=trajectory.iterations,
            simulation=trajectory.simulation,
        ),
    )

    result = calculate_hbn_reference_polarization(
        data,
        HBNReferencePolarizationRequest(
            reference_path=reference_path,
            replication=(2, 1, 1),
            charge_source="reaxff",
            volume_method="cell",
        ),
    )

    assert result.table.iloc[0]["charge_source"] == "reaxff"
    assert result.table.iloc[0]["dipole_c (e*angstrom)"] == pytest.approx(-8 * 4.0 * 0.1)
    assert set(result.displacements["charge (e)"]) == {-4.0, 4.0}


def test_replication_must_match_trajectory_atom_count(reference_path) -> None:
    trajectory = _trajectory_from_reference(reference_path)

    with pytest.raises(ValueError, match="produces 8 reference atoms.*contains 16"):
        calculate_hbn_reference_polarization(
            trajectory,
            HBNReferencePolarizationRequest(
                reference_path=reference_path, replication=(1, 1, 1)
            ),
        )


@pytest.fixture
def reference_path():
    return REFERENCE_STRUCTURE_PATH
