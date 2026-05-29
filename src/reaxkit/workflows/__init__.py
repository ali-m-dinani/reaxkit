"""Workflow utilities for ``__init__``.

This module defines CLI task wiring and helper routines for ReaxKit workflows.

**Usage context**

- Command routing: Resolve CLI aliases and normalized command names.
- Task execution: Build request objects and invoke registered tasks.
- Output handling: Forward results to table, plot, export, or report flows.
"""

from reaxkit.core.results_shaping.result_bundle import (
    DualTableResultBundle,
    bundle_canonical_and_tract_tables,
)

__all__ = ["DualTableResultBundle", "bundle_canonical_and_tract_tables"]
