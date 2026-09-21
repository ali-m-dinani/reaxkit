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
from reaxkit.core.platform.constants import const
from reaxkit.domain.data_models import TrajectoryData


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
