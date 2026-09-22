from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from reaxkit.analysis.ferroelectrics.hbn_reference.polarization import (
    HBNReferencePolarizationRequest,
    REFERENCE_STRUCTURE_PATH,
    calculate_hbn_reference_polarization,
)
from reaxkit.analysis.ferroelectrics.hbn_reference.local_polarization import (
    HBNReferenceLocalPolarizationRequest,
    calculate_hbn_reference_local_polarization,
)
from reaxkit.analysis.ferroelectrics.hbn_reference.projected_polarity import (
    HBNReferenceProjectedPolarityRequest,
    calculate_hbn_reference_projected_polarity,
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


def test_reference_tracks_neutral_primitive_cells(reference_path) -> None:
    trajectory = _trajectory_from_reference(reference_path)
    result = calculate_hbn_reference_polarization(
        trajectory,
        HBNReferencePolarizationRequest(
            reference_path=reference_path, replication=(2, 1, 1)
        ),
    )

    counts = result.displacements.groupby("local_cell_id").size()
    assert len(counts) == 4
    assert set(counts) == {4}
    for _, group in result.displacements.groupby("local_cell_id"):
        assert sorted(group["reference_element"].tolist()) == ["Al", "Al", "N", "N"]


def test_local_equal_volumes_close_to_global_dipole_and_polarization(reference_path) -> None:
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
    result = calculate_hbn_reference_local_polarization(
        trajectory,
        HBNReferenceLocalPolarizationRequest(
            reference_path=reference_path,
            replication=(2, 1, 1),
            volume_method="cell",
        ),
    )

    assert len(result.table) == 4
    assert set(result.table["atom_count"]) == {4}
    assert set(result.table["composition"]) == {"Al2,N2"}
    assert result.table["dipole_z (e*angstrom)"].to_numpy() == pytest.approx(
        np.full(4, -0.6)
    )
    frame_volume = result.reference_result.table.iloc[0]["volume (angstrom^3)"]
    assert result.table["local_volume (angstrom^3)"].sum() == pytest.approx(frame_volume)
    summary = result.summary.iloc[0]
    assert summary["dipole_z (e*angstrom)"] == pytest.approx(
        result.reference_result.table.iloc[0]["dipole_z (e*angstrom)"]
    )
    assert summary["dipole_closure_error_z (e*angstrom)"] == pytest.approx(0.0)
    assert summary["P_z (uC/cm^2)"] == pytest.approx(
        result.reference_result.table.iloc[0]["P_z (uC/cm^2)"]
    )


def test_local_deformation_volumes_are_normalized_to_selected_frame_volume(
        reference_path,
) -> None:
    trajectory = _trajectory_from_reference(reference_path)
    result = calculate_hbn_reference_local_polarization(
        trajectory,
        HBNReferenceLocalPolarizationRequest(
            reference_path=reference_path,
            replication=(2, 1, 1),
            local_volume_method="deformation",
            volume_method="hull",
        ),
    )

    assert np.all(result.table["local_volume (angstrom^3)"] > 0.0)
    assert result.table["local_volume (angstrom^3)"].sum() == pytest.approx(
        result.reference_result.table.iloc[0]["volume (angstrom^3)"]
    )


def test_projected_polarity_includes_zero_and_uses_fixed_cell_bins(reference_path) -> None:
    single = _trajectory_from_reference(reference_path)
    labels = np.asarray(single.elements)
    positions = np.repeat(single.positions, 2, axis=0)
    positions[1, labels == "Al", 2] += 0.1
    assert single.simulation is not None
    simulation = SimulationData(
        atom_ids=single.atom_ids,
        iterations=np.asarray([20, 40]),
        elements=single.elements,
        cell_lengths=np.repeat(single.simulation.cell_lengths, 2, axis=0),
        cell_angles=np.repeat(single.simulation.cell_angles, 2, axis=0),
    )
    trajectory = TrajectoryData(
        positions=positions,
        elements=single.elements,
        atom_ids=single.atom_ids,
        iterations=np.asarray([20, 40]),
        simulation=simulation,
    )
    result = calculate_hbn_reference_projected_polarity(
        trajectory,
        HBNReferenceProjectedPolarityRequest(
            reference_path=reference_path,
            replication=(2, 1, 1),
            frames=(0, 1),
            component="z",
            projection_plane="xz",
            projection_bins=(1, 1),
            profile_axis="z",
        ),
    )

    projected = result.projected_bins.set_index("frame_index")
    assert projected.loc[0, "defined_cell_count"] == 4
    assert projected.loc[0, "zero_count"] == 4
    assert projected.loc[0, "mean_polarity"] == pytest.approx(0.0)
    assert projected.loc[1, "negative_count"] == 4
    assert projected.loc[1, "mean_polarity"] == pytest.approx(-1.0)
    assert result.centers.groupby("local_cell_id")["u_bin"].nunique().max() == 1
    assert result.centers.groupby("local_cell_id")["v_bin"].nunique().max() == 1
    assert len(result.kymograph_bins) == 2


def test_reaxff_local_cells_are_neutralized_without_losing_raw_global_closure(
        reference_path,
) -> None:
    base = _trajectory_from_reference(reference_path)
    mapping_result = calculate_hbn_reference_polarization(
        base,
        HBNReferencePolarizationRequest(
            reference_path=reference_path, replication=(2, 1, 1)
        ),
    )
    mapping = mapping_result.mapping.sort_values("atom_index")
    cell_ids = mapping["local_cell_id"].to_numpy(int)
    positions = base.positions.copy()
    cell_translation = np.where(cell_ids < 2, 0.05, -0.05)
    positions[0, :, 2] += cell_translation
    trajectory = TrajectoryData(
        positions=positions,
        elements=base.elements,
        atom_ids=base.atom_ids,
        iterations=base.iterations,
        simulation=base.simulation,
    )
    labels = np.asarray(base.elements)
    formal = np.where(labels == "Al", 3.0, -3.0)
    charge_offset = np.where(cell_ids < 2, 0.2, -0.2)
    data = ElectrostaticsData(
        trajectory=trajectory,
        charges=ChargeData(
            charges=(formal + charge_offset)[None, :],
            iterations=base.iterations,
            simulation=base.simulation,
        ),
    )

    result = calculate_hbn_reference_local_polarization(
        data,
        HBNReferenceLocalPolarizationRequest(
            reference_path=reference_path,
            replication=(2, 1, 1),
            charge_source="reaxff",
            volume_method="cell",
        ),
    )

    assert result.table["charge_neutralization_applied"].all()
    assert result.table["effective_net_charge (e)"].to_numpy() == pytest.approx(
        np.zeros(4), abs=1.0e-12
    )
    assert result.table["dipole_z (e*angstrom)"].to_numpy() == pytest.approx(
        np.zeros(4), abs=1.0e-12
    )
    summary = result.summary.iloc[0]
    assert summary["raw_dipole_closure_error_z (e*angstrom)"] == pytest.approx(
        0.0, abs=1.0e-12
    )
    assert summary["raw_dipole_z (e*angstrom)"] == pytest.approx(
        result.reference_result.table.iloc[0]["dipole_z (e*angstrom)"]
    )


@pytest.fixture
def reference_path():
    return REFERENCE_STRUCTURE_PATH
