from __future__ import annotations

import numpy as np
import pytest

from reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.dipole import (
    BasalPlaneDipoleRequest,
    calculate_basal_plane_dipoles,
)
from reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.polarization import (
    BasalPlanePolarizationRequest,
    calculate_basal_plane_polarization,
)
from reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.local_polarization import (
    BasalPlaneLocalPolarizationRequest,
    calculate_basal_plane_local_polarization,
    write_local_polarization_extxyz,
)
from reaxkit.core.platform.constants import const
from reaxkit.domain.data_models import (
    ChargeData,
    ElectricFieldData,
    ElectrostaticsData,
    TrajectoryData,
)


def _trajectory(*, include_apical: bool = True) -> TrajectoryData:
    positions = [
        [0.0, 0.0, 0.0],
        [1.6, 0.0, -0.2],
        [-0.8, 1.386, -0.2],
        [-0.8, -1.386, -0.2],
    ]
    if include_apical:
        positions.append([0.0, 0.0, 1.8])
    return TrajectoryData(
        positions=np.asarray([positions], dtype=float),
        elements=["Al", *("N" for _ in positions[1:])],
        atom_ids=list(range(1, len(positions) + 1)),
        iterations=np.asarray([10]),
    )


def _dipole_request() -> BasalPlaneDipoleRequest:
    return BasalPlaneDipoleRequest(
        periodic=(False, False, False),
        charge_source="formal",
        formal_charges={"Al": 3.0, "N": -3.0},
    )


def test_dipole_uses_fixed_anion_reference_and_center_displacement() -> None:
    result = calculate_basal_plane_dipoles(_trajectory(), _dipole_request())
    site = result.table.iloc[0]

    assert bool(site["has_basal_plane_dipole"])
    assert site["basal_mean_z (angstrom)"] == pytest.approx(-0.2)
    assert site["center_displacement_z (angstrom)"] == pytest.approx(0.2)
    assert site["apical_displacement_z (angstrom)"] == pytest.approx(2.0)
    assert site["center_mu_z (e*angstrom)"] == pytest.approx(-0.6)
    assert site["basal_mu_z (e*angstrom)"] == pytest.approx(0.0)
    assert site["apical_mu_z (e*angstrom)"] == pytest.approx(0.0)
    assert site["mu_z (e*angstrom)"] == pytest.approx(-0.6)
    assert site["included_ion_count"] == 1
    assert site["electron_charge_sign"] == -1.0
    assert result.ions["atom_id"].tolist() == [1, 2, 3, 4, 5]
    assert result.summary.iloc[0]["ion_count"] == 5
    assert result.summary.iloc[0]["displaced_ion_count"] == 1
    fixed = result.ions[result.ions["role"] == "fixed_reference"]
    assert np.allclose(fixed["displacement_z (angstrom)"], 0.0)


def test_missing_apical_still_uses_center_displacement() -> None:
    result = calculate_basal_plane_dipoles(
        _trajectory(include_apical=False), _dipole_request()
    )
    site = result.table.iloc[0]

    assert bool(site["has_basal_plane_dipole"])
    assert site["included_ion_count"] == 1
    assert site["mu_z (e*angstrom)"] == pytest.approx(-0.6)


def test_polarization_uses_new_dipole_and_cell_volume() -> None:
    request = BasalPlanePolarizationRequest(
        periodic=(False, False, False),
        charge_source="formal", formal_charges={"Al": 3.0, "N": -3.0},
        cell_lengths=(10.0, 10.0, 10.0),
        bins_x=1, bins_y=1, bins_z=1, volume_method="cell",
    )
    result = calculate_basal_plane_polarization(_trajectory(), request)
    row = result.table.iloc[0]

    assert row["mu_z (e*angstrom)"] == pytest.approx(-0.6)
    assert row["volume (angstrom^3)"] == pytest.approx(1000.0)
    expected = -0.6 / 1000.0 * float(const("ea3_to_uC_cm2"))
    assert row["P_z (uC/cm^2)"] == pytest.approx(expected)


def test_local_polarization_defaults_to_equal_frame_volume_shares() -> None:
    first = np.asarray(_trajectory().positions[0], dtype=float)
    second = first + np.asarray([6.0, 0.0, 0.0])
    positions = np.concatenate((first, second), axis=0)
    trajectory = TrajectoryData(
        positions=positions[None, :, :],
        elements=["Al", "N", "N", "N", "N"] * 2,
        atom_ids=list(range(1, 11)),
        iterations=np.asarray([10]),
    )
    request = BasalPlaneLocalPolarizationRequest(
        periodic=(False, False, False),
        charge_source="formal",
        formal_charges={"Al": 3.0, "N": -3.0},
        cell_lengths=(20.0, 10.0, 10.0),
        volume_method="cell",
    )

    result = calculate_basal_plane_local_polarization(trajectory, request)

    assert request.local_volume_method == "equal"
    assert result.table["valid_center_count"].tolist() == [2, 2]
    assert result.table["local_volume (angstrom^3)"].tolist() == pytest.approx(
        [1000.0, 1000.0]
    )
    expected = -0.6 / 1000.0 * float(const("ea3_to_uC_cm2"))
    assert result.table["P_z (uC/cm^2)"].tolist() == pytest.approx(
        [expected, expected]
    )
    assert result.summary.iloc[0]["assigned_volume (angstrom^3)"] == pytest.approx(
        2000.0
    )


def test_local_polarization_supports_coordination_tetrahedron_volume() -> None:
    request = BasalPlaneLocalPolarizationRequest(
        periodic=(False, False, False),
        charge_source="formal",
        formal_charges={"Al": 3.0, "N": -3.0},
        local_volume_method="coordination",
    )

    result = calculate_basal_plane_local_polarization(_trajectory(), request)
    row = result.table.iloc[0]

    assert bool(row["has_valid_local_volume"])
    assert row["local_volume (angstrom^3)"] == pytest.approx(2.2176)
    expected = -0.6 / 2.2176 * float(const("ea3_to_uC_cm2"))
    assert row["P_z (uC/cm^2)"] == pytest.approx(expected)


def test_coordination_volume_marks_missing_apical_neighbor_invalid() -> None:
    request = BasalPlaneLocalPolarizationRequest(
        periodic=(False, False, False),
        charge_source="formal",
        formal_charges={"Al": 3.0, "N": -3.0},
        local_volume_method="coordination",
    )

    row = calculate_basal_plane_local_polarization(
        _trajectory(include_apical=False), request
    ).table.iloc[0]

    assert not bool(row["has_valid_local_volume"])
    assert np.isnan(row["local_volume (angstrom^3)"])
    assert np.isnan(row["P_z (uC/cm^2)"])


def test_equal_volume_defaults_to_occupied_hull_excluding_vacuum() -> None:
    request = BasalPlaneLocalPolarizationRequest(
        periodic=(False, False, False),
        charge_source="formal",
        formal_charges={"Al": 3.0, "N": -3.0},
        cell_lengths=(10.0, 10.0, 100.0),
    )

    row = calculate_basal_plane_local_polarization(_trajectory(), request).table.iloc[0]

    assert request.volume_method == "hull"
    assert row["frame_volume (angstrom^3)"] == pytest.approx(2.2176)
    assert row["local_volume (angstrom^3)"] == pytest.approx(2.2176)
    assert row["frame_volume (angstrom^3)"] < 10.0 * 10.0 * 100.0


def test_local_polarization_extended_xyz_contains_ovito_vectors(tmp_path) -> None:
    request = BasalPlaneLocalPolarizationRequest(
        periodic=(False, False, False),
        charge_source="formal",
        formal_charges={"Al": 3.0, "N": -3.0},
        cell_lengths=(10.0, 10.0, 10.0),
        volume_method="cell",
    )
    result = calculate_basal_plane_local_polarization(_trajectory(), request)
    path = write_local_polarization_extxyz(result, tmp_path / "local.extxyz")
    text = path.read_text(encoding="utf-8")

    assert "is_local_center:I:1" in text
    assert "has_valid_local_volume:I:1" in text
    assert "local_volume:R:1" in text
    assert "local_dipole:R:3" in text
    assert "local_polarization:R:3" in text
    assert "local_volume_method=equal" in text


def test_local_extxyz_matches_restored_source_frame_rows(tmp_path) -> None:
    original = _trajectory()
    trajectory = TrajectoryData(
        positions=np.repeat(original.positions, 2, axis=0),
        elements=original.elements,
        atom_ids=original.atom_ids,
        iterations=np.asarray([10, 20]),
    )
    request = BasalPlaneLocalPolarizationRequest(
        periodic=(False, False, False),
        charge_source="formal",
        formal_charges={"Al": 3.0, "N": -3.0},
        cell_lengths=(10.0, 10.0, 10.0),
        volume_method="cell",
    )
    result = calculate_basal_plane_local_polarization(trajectory, request)
    trajectory.source_frame_indices = np.asarray([0, 50])
    for table in (result.table, result.summary, result.dipole_result.table):
        table.loc[table["frame_index"].eq(1), "frame_index"] = 50

    path = write_local_polarization_extxyz(result, tmp_path / "selected.extxyz")
    lines = path.read_text(encoding="utf-8").splitlines()
    second_header = lines[8]
    second_al = lines[9].split()

    assert "frame=50" in second_header
    assert "iter=20" in second_header
    assert second_al[5:7] == ["1", "1"]
    assert second_al[7] != "nan"


def test_local_extxyz_adds_only_iteration_aligned_field_metadata(tmp_path) -> None:
    trajectory = _trajectory()
    data = ElectrostaticsData(
        trajectory=trajectory,
        charges=ChargeData(
            charges=np.zeros((1, 5)),
            iterations=np.asarray([10]),
            simulation=trajectory.simulation,
        ),
        electric_field=ElectricFieldData(
            applied_field_values=np.asarray([[0.25]]),
            applied_field_components=("field_z",),
            sampled_field_iterations=np.asarray([10]),
        ),
    )
    request = BasalPlaneLocalPolarizationRequest(
        periodic=(False, False, False),
        charge_source="formal",
        formal_charges={"Al": 3.0, "N": -3.0},
        cell_lengths=(10.0, 10.0, 10.0),
        volume_method="cell",
        include_electric_field=True,
        field_direction="z",
    )

    result = calculate_basal_plane_local_polarization(data, request)
    path = write_local_polarization_extxyz(result, tmp_path / "field.extxyz")
    header = path.read_text(encoding="utf-8").splitlines()[1]
    converted = 0.25 * float(const("electric_field_VA_to_MVcm"))

    assert f"electric_field={converted}" in header
    assert "electric_field_direction=z" in header
    assert "electric_field_units=MV/cm" in header
    assert "charge:R:1" not in header
    assert "delta_charge" not in header
