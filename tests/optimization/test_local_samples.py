"""Optional read-only checks against installed optimizer checkouts."""

import os
from pathlib import Path

import pytest

from reaxkit.optimization.conversion import collect_native

SAMPLES = (
    ("jax_soppi", "JAX-SOPPI/outputs/smoke_full", "progress_count"),
    ("jax_reaxff", "JAX-ReaxFF/ffields", "prediction_count"),
    ("indeedopt", "INDEEDopt/short_run", "progress_count"),
    ("ogolem", "ogolem/converted/cobalt_jax", "progress_count"),
    ("pibo", "pibo_mos2/results/cobalt_quick_pibo", "progress_count"),
    ("ezff", "EZFF/examples/reaxff-charge-gulp-serial/smoke-lammps-output", "progress_count"),
)


@pytest.mark.parametrize("optimizer,relative,field", SAMPLES)
def test_existing_local_reports(tmp_path, optimizer, relative, field):
    root = os.environ.get("REAXKIT_OPTIMIZER_CHECKOUTS")
    if not root:
        pytest.skip("Set REAXKIT_OPTIMIZER_CHECKOUTS for read-only local sample validation")
    source = Path(root) / relative
    if not source.is_dir():
        pytest.skip(f"Local sample unavailable: {relative}")
    result = collect_native(optimizer, source, tmp_path / "normalized")
    assert result[field] > 0, result
