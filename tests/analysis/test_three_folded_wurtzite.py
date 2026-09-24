from __future__ import annotations

import numpy as np

from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.neighbors import (
    WurtziteNeighborRequest,
    extract_wurtzite_neighbors,
)
from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.polarity import (
    WurtzitePolarityRequest,
    calculate_wurtzite_polarity,
)
from reaxkit.domain.data_models import TrajectoryData


def _trajectory(neighbor_positions: list[list[float]]) -> TrajectoryData:
    positions = np.asarray([[[0.0, 0.0, 0.0], *neighbor_positions]], dtype=float)
    elements = ["Al", *("N" for _ in neighbor_positions)]
    return TrajectoryData(
        positions=positions,
        elements=elements,
        atom_ids=list(range(1, len(elements) + 1)),
        iterations=np.asarray([0]),
    )


def _request() -> WurtzitePolarityRequest:
    return WurtzitePolarityRequest(
        periodic=(False, False, False),
        charge_source="formal",
        formal_charges={"Al": 3.0, "N": -3.0},
    )


def test_exactly_three_basal_neighbors_produce_known_properties() -> None:
    trajectory = _trajectory([
        [1.6, 0.0, -0.2],
        [-0.8, 1.386, -0.2],
        [-0.8, -1.386, -0.2],
    ])
    request = _request()
    neighbors = extract_wurtzite_neighbors(trajectory, request)
    result = calculate_wurtzite_polarity(neighbors, request)
    site = result.table.iloc[0]

    assert bool(site["has_three_basal_neighbors"])
    assert not bool(site["has_apical_neighbor"])
    assert np.isclose(site["delta (angstrom)"], 0.2)
    assert site["polarity"] == 1
    assert site["polarity_label"] == "UP"
    assert site["eta_c (e*angstrom)"] > 0.0
    assert result.basal_neighbors["neighbor_atom_id"].tolist() == [2, 3, 4]


def test_up_polarity_selects_upper_apical_candidate_and_ignores_lower_one() -> None:
    trajectory = _trajectory([
        [1.6, 0.0, -0.2],
        [-0.8, 1.386, -0.2],
        [-0.8, -1.386, -0.2],
        [0.0, 0.0, -1.7],
        [0.0, 0.0, 1.8],
    ])
    request = _request()
    result = calculate_wurtzite_polarity(
        extract_wurtzite_neighbors(trajectory, request), request
    )
    site = result.table.iloc[0]

    assert site["polarity_label"] == "UP"
    assert bool(site["has_apical_neighbor"])
    assert site["apical_neighbor_atom_id"] == 6
    assert np.isclose(site["apical_bond_c (angstrom)"], 1.8)
    assert result.ignored_neighbors["neighbor_atom_id"].tolist() == [5]


def test_opposite_side_candidate_is_not_used_as_apical() -> None:
    trajectory = _trajectory([
        [1.6, 0.0, -0.2],
        [-0.8, 1.386, -0.2],
        [-0.8, -1.386, -0.2],
        [0.0, 0.0, -1.7],
    ])
    request = _request()
    result = calculate_wurtzite_polarity(
        extract_wurtzite_neighbors(trajectory, request), request
    )
    site = result.table.iloc[0]

    assert site["polarity_label"] == "UP"
    assert not bool(site["has_apical_neighbor"])
    assert site["apical_neighbor_atom_id"] == -1
    assert result.apical_neighbors.empty
    assert result.ignored_neighbors["neighbor_atom_id"].tolist() == [5]
