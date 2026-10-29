import numpy as np
import pytest
from scipy.spatial import cKDTree

from reaxkit.analysis.electrostatics.potential_and_electric_field.calculation import _tree_pairs


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("translation", [[0, 0, 0], [10, 0, 0], [100, 100, 100], [-3, 5, 1]])
def test_pruned_pairs_equal_unpruned_query(sign, translation):
    positions = np.random.default_rng(19).uniform(-5, 5, (150, 3))
    tree = cKDTree(positions)
    targets = np.arange(0, len(positions), 2)
    translation = np.asarray(translation)
    neighbors = tree.query_ball_point(positions[targets] + sign * translation, 3.0)
    expected = [(int(target), int(source)) for target, sources in zip(targets, neighbors) for source in sources]
    actual_targets, actual_sources = _tree_pairs(tree, positions, targets, translation, 3.0, translation_sign=sign)
    assert list(zip(actual_targets, actual_sources)) == expected


def test_cutoff_boundary_is_kept():
    positions = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    tree = cKDTree(positions)
    targets, sources = _tree_pairs(tree, positions, np.asarray([1]), np.asarray([3.0, 0.0, 0.0]),
                                  3.0, translation_sign=1.0)
    assert list(zip(targets, sources)) == [(1, 1)]


def test_disjoint_image_does_not_query_tree():
    class NoQueryTree:
        mins = np.zeros(3)
        maxes = np.ones(3)

        def query_ball_point(self, *args, **kwargs):
            raise AssertionError("Disjoint image must not launch a neighbor query")

    targets, sources = _tree_pairs(NoQueryTree(), np.zeros((1, 3)), np.asarray([0]),
                                  np.full(3, 100.0), 10.0, translation_sign=1.0)
    assert targets.size == sources.size == 0
