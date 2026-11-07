"""Load adapter contracts without importing optimizer libraries."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from pathlib import Path

import yaml

from .models import OptimizationError

OPTIMIZERS = (
    "ams_mcff", "ams_cmaes", "standalone_soppi", "jax_soppi",
    "jax_reaxff", "indeedopt", "ogolem", "pibo", "ezff",
)


@lru_cache(maxsize=None)
def specification(optimizer: str) -> dict:
    if optimizer not in OPTIMIZERS:
        raise OptimizationError(f"Unknown optimizer {optimizer!r}; choose from {', '.join(OPTIMIZERS)}")
    path = Path(__file__).parent / "adapters" / optimizer / "adapter.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def converter(optimizer: str):
    specification(optimizer)
    return import_module(f"reaxkit.optimization.adapters.{optimizer}").convert
