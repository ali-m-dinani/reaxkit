from __future__ import annotations

import numpy as np
import pytest

from reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.projected_polarity import (
    BasalPlaneProjectedPolarityRequest,
    calculate_basal_plane_projected_polarity,
)
from reaxkit.domain.data_models import TrajectoryData


def _motif(offset: float, *, inverted: bool) -> list[list[float]]:
    basal_z = 0.2 if inverted else -0.2
    apical_z = -1.8 if inverted else 1.8
    return [
        [offset, 0.0, 0.0],
        [offset + 1.6, 0.0, basal_z],
        [offset - 0.8, 1.386, basal_z],
        [offset - 0.8, -1.386, basal_z],
        [offset, 0.0, apical_z],
    ]


def _trajectory() -> TrajectoryData:
    mixed = _motif(0.0, inverted=False) + _motif(6.0, inverted=True)
    negative = _motif(0.0, inverted=False) + _motif(-6.0, inverted=False)
    return TrajectoryData(
        positions=np.asarray([mixed, negative], dtype=float),
        elements=["Al", "N", "N", "N", "N"] * 2,
        atom_ids=list(range(1, 11)),
        iterations=np.asarray([0, 50]),
    )


def _request() -> BasalPlaneProjectedPolarityRequest:
    return BasalPlaneProjectedPolarityRequest(
        periodic=(False, False, False),
        charge_source="formal",
        formal_charges={"Al": 3.0, "N": -3.0},
        projection_plane="xz",
        projection_bins=(1, 1),
        profile_axis="z",
    )


def test_projected_polarity_is_mean_of_center_signs() -> None:
    result = calculate_basal_plane_projected_polarity(_trajectory(), _request())
    first = result.projected_bins[result.projected_bins["frame_index"].eq(0)].iloc[0]
    second = result.projected_bins[result.projected_bins["frame_index"].eq(1)].iloc[0]

    assert sorted(result.centers[result.centers["frame_index"].eq(0)]["polarity"]) == [
        -1.0, 1.0
    ]
    assert first["defined_center_count"] == 2
    assert first["positive_count"] == 1
    assert first["negative_count"] == 1
    assert first["positive_fraction"] == pytest.approx(0.5)
    assert first["negative_fraction"] == pytest.approx(0.5)
    assert first["mean_polarity"] == pytest.approx(0.0)
    assert second["mean_polarity"] == pytest.approx(-1.0)


def test_kymograph_profile_uses_same_unweighted_polarity_mean() -> None:
    result = calculate_basal_plane_projected_polarity(_trajectory(), _request())

    assert result.kymograph_bins["mean_polarity"].tolist() == pytest.approx([0.0, -1.0])
    assert result.kymograph_bins["defined_center_count"].tolist() == [2, 2]


def test_reference_frame_fixes_atom_bin_membership_across_frames() -> None:
    request = _request()
    request.projection_bins = (2, 1)
    request.frames = [1]

    result = calculate_basal_plane_projected_polarity(_trajectory(), request)
    moved_center = result.centers[result.centers["site_atom_id"].eq(6)].iloc[0]

    assert moved_center["site_x (angstrom)"] == pytest.approx(-6.0)
    assert moved_center["reference_u (angstrom)"] == pytest.approx(6.0)
    assert moved_center["u_bin"] == 1
    assert result.projected_bins.loc[
               result.projected_bins["u_bin"].eq(1), "defined_center_count"
           ].iloc[0] == 1


def test_zero_dipole_is_included_as_zero_polarity() -> None:
    positions = _motif(0.0, inverted=False)
    for index in (1, 2, 3):
        positions[index][2] = 0.0
    trajectory = TrajectoryData(
        positions=np.asarray([positions], dtype=float),
        elements=["Al", "N", "N", "N", "N"],
        atom_ids=list(range(1, 6)),
        iterations=np.asarray([0]),
    )

    result = calculate_basal_plane_projected_polarity(trajectory, _request())
    row = result.projected_bins.iloc[0]

    assert result.centers.iloc[0]["polarity"] == 0.0
    assert row["defined_center_count"] == 1
    assert row["zero_count"] == 1
    assert row["zero_fraction"] == pytest.approx(1.0)
    assert row["mean_polarity"] == pytest.approx(0.0)


def test_profile_axis_must_belong_to_projection_plane() -> None:
    request = _request()
    request.profile_axis = "y"

    with pytest.raises(ValueError, match="projection-plane axes"):
        calculate_basal_plane_projected_polarity(_trajectory(), request)
