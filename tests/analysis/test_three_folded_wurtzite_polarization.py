from __future__ import annotations

import numpy as np
import pytest

from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.polarization import (
    BinnedPolarizationRequest,
    calculate_binned_polarization,
)
from reaxkit.core.platform.constants import const
from reaxkit.domain.data_models import TrajectoryData


def _trajectory() -> TrajectoryData:
    return TrajectoryData(
        positions=np.asarray([[
            [0.0, 0.0, 0.0],
            [1.6, 0.0, -0.2],
            [-0.8, 1.386, -0.2],
            [-0.8, -1.386, -0.2],
        ]]),
        elements=["Al", "N", "N", "N"],
        atom_ids=[1, 2, 3, 4],
        iterations=np.asarray([25]),
    )


@pytest.mark.parametrize("method", ["hull", "bbox", "cell"])
def test_binned_polarization_supports_all_volume_methods(method: str) -> None:
    request = BinnedPolarizationRequest(
        periodic=(False, False, False),
        charge_source="formal",
        formal_charges={"Al": 3.0, "N": -3.0},
        cell_lengths=(10.0, 10.0, 10.0),
        bins_x=1, bins_y=1, bins_z=1,
        volume_method=method,
    )
    result = calculate_binned_polarization(_trajectory(), request)

    row = result.table.iloc[0]
    assert row["valid_dipole_count"] == 1
    assert row["mu_z (e*angstrom)"] == pytest.approx(1.8)
    assert row["volume (angstrom^3)"] > 0.0
    assert np.isfinite(row["P_z (uC/cm^2)"])


def test_cell_volume_is_divided_between_requested_directional_bins() -> None:
    request = BinnedPolarizationRequest(
        periodic=(False, False, False),
        charge_source="formal",
        formal_charges={"Al": 3.0, "N": -3.0},
        cell_lengths=(10.0, 10.0, 10.0),
        bins_x=2, bins_y=1, bins_z=1,
        volume_method="cell",
    )
    result = calculate_binned_polarization(_trajectory(), request)

    assert len(result.table) == 2
    assert result.table["volume (angstrom^3)"].tolist() == pytest.approx([500.0, 500.0])
    summary = result.summary.iloc[0]
    expected = 1.8 / 1000.0 * float(const("ea3_to_uC_cm2"))
    assert summary["P_z (uC/cm^2)"] == pytest.approx(expected)
