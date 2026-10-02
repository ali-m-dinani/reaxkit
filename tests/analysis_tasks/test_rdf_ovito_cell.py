from __future__ import annotations

import numpy as np
import pytest

from reaxkit.analysis.trajectory.rdf import RDFRequest, RDFTask
from reaxkit.domain.data_models import SimulationData, TrajectoryData


def _trajectory(positions, length=10.0):
    positions = np.asarray(positions, dtype=float)[None, :, :]
    atom_ids = list(range(1, positions.shape[1] + 1))
    return TrajectoryData(
        positions=positions, atom_ids=atom_ids, elements=["He"] * len(atom_ids),
        simulation=SimulationData(atom_ids=atom_ids, cell_lengths=np.full((1, 3), length)),
    )


def test_ovito_counts_periodic_neighbors_with_correct_volume():
    pytest.importorskip("ovito")
    data = _trajectory([[0.1, 5, 5], [9.9, 5, 5]])
    result = RDFTask().run(data, RDFRequest(backend="ovito", bins=10, r_max=1.0))
    edges = np.linspace(0, 1, 11)
    shell_volumes = 4 * np.pi / 3 * np.diff(edges ** 3)
    coordination = np.sum(result.table["g"].to_numpy() * shell_volumes * 2 / 1000)
    assert coordination == pytest.approx(1.0)
    assert result.table.loc[result.table["g"].idxmax(), "r"] < 0.3


def test_ovito_normalization_uses_simulation_volume():
    pytest.importorskip("ovito")
    request = RDFRequest(backend="ovito", bins=10, r_max=2)
    positions = [[4, 5, 5], [5, 5, 5]]
    small = RDFTask().run(_trajectory(positions, 10), request)
    large = RDFTask().run(_trajectory(positions, 20), request)
    np.testing.assert_allclose(large.table["g"], 8 * small.table["g"])


def test_ovito_identical_selected_groups_and_disjoint_groups():
    pytest.importorskip("ovito")
    data = _trajectory([[0.1, 5, 5], [9.9, 5, 5], [5, 5, 5]])
    for atom_ids_a, atom_ids_b in (([1, 2], [1, 2]), ([1], [2])):
        result = RDFTask().run(data, RDFRequest(backend="ovito", bins=10, r_max=1, atom_ids_a=atom_ids_a, atom_ids_b=atom_ids_b))
        assert result.table["g"].max() > 0
    with pytest.raises(ValueError, match="identical or disjoint"):
        RDFTask().run(data, RDFRequest(backend="ovito", atom_ids_a=[1, 2], atom_ids_b=[2, 3]))


def test_ovito_requires_cell():
    pytest.importorskip("ovito")
    data = _trajectory([[1, 1, 1], [2, 1, 1]])
    data.simulation = None
    with pytest.raises(ValueError, match="cell_lengths is required"):
        RDFTask().run(data, RDFRequest(backend="ovito"))
