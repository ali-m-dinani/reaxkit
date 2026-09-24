"""Backward-compatible result-bundle imports."""

from reaxkit.core.results_shaping.result_bundle import (
    DualTableResultBundle,
    bundle_canonical_and_tract_tables,
)

__all__ = ["DualTableResultBundle", "bundle_canonical_and_tract_tables"]
