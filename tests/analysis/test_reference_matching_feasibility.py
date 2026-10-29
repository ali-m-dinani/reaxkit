import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix

from reaxkit.analysis.ferroelectrics.hbn_reference import polarization


@pytest.mark.parametrize("seed", range(12))
def test_reference_matching_preserves_minimum_cost(seed):
    random = np.random.default_rng(seed)
    costs = random.uniform(0.1, 10, size=(24, 24))
    allowed = random.random(costs.shape) < 0.2
    allowed[np.arange(24), random.permutation(24)] = True
    graph = csr_matrix(np.where(allowed, costs, 0))
    original = graph.copy()
    expected_rows, expected_columns = linear_sum_assignment(np.where(allowed, costs, np.inf))
    rows, columns = polarization._minimum_weight_reference_matching(graph)
    assert len(np.unique(columns)) == 24
    assert allowed[rows, columns].all()
    assert costs[rows, columns].sum() == pytest.approx(costs[expected_rows, expected_columns].sum())
    assert (graph != original).nnz == 0


def test_infeasible_graph_does_not_enter_weighted_solver(monkeypatch):
    def unexpected_solver(graph):
        pytest.fail("Infeasible graph reached weighted solver")

    monkeypatch.setattr(polarization, "min_weight_full_bipartite_matching", unexpected_solver)
    graph = csr_matrix([[1, 0, 0], [2, 0, 0], [0, 3, 4]])
    with pytest.raises(ValueError, match="no full"):
        polarization._minimum_weight_reference_matching(graph)


def test_feasibility_seed_preserves_edges_and_weights(monkeypatch):
    graph = csr_matrix([[1, 2, 0], [1, 0, 3], [0, 1, 2]])
    solver = polarization.min_weight_full_bipartite_matching

    def inspect_seed(ordered):
        assert len(np.unique(ordered.indices[ordered.indptr[:-1]])) == 3
        np.testing.assert_array_equal(ordered.toarray(), graph.toarray())
        return solver(ordered)

    monkeypatch.setattr(polarization, "min_weight_full_bipartite_matching", inspect_seed)
    polarization._minimum_weight_reference_matching(graph)
