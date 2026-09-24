"""Regression tests for the direction of polar surfaces."""

from pathlib import Path

import numpy as np
import pytest

from reaxkit.engine.common.io.geo_io import read_structure
from reaxkit.engine.common.generators.structure_transformers import build_surface


@pytest.mark.parametrize("vacuum", [0, 10])
def test_opposite_aln_miller_indices_expose_opposite_terminations(vacuum):
    cif = Path(__file__).resolve().parents[4] / "examples_to_test" / "AlN_ortho.cif"
    bulk = read_structure(cif)

    al_polar = build_surface(bulk, (0, 0, 1), layers=8, vacuum=vacuum)
    n_polar = build_surface(bulk, (0, 0, -1), layers=8, vacuum=vacuum)

    def top_species(slab):
        z = slab.positions[:, 2]
        return set(np.asarray(slab.get_chemical_symbols())[np.isclose(z, z.max())])

    assert top_species(al_polar) == {"Al"}
    assert top_species(n_polar) == {"N"}
    np.testing.assert_allclose(al_polar.cell, n_polar.cell)
    np.testing.assert_allclose(
        np.sort(al_polar.positions[:, 2]),
        np.sort(n_polar.cell[2, 2] - n_polar.positions[:, 2]),
        atol=1e-12,
    )
