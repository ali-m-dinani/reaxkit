"""Helpers for packaging multi-table analysis outputs for presentation/persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class DualTableResultBundle:
    """Result container with stable field names used by persistence."""

    table: pd.DataFrame
    tract_table: pd.DataFrame
    summary: dict[str, Any] | None = None
    request: Any = None


def bundle_canonical_and_tract_tables(
    result: Any,
    *,
    canonical_attr: str = "table",
    tract_attr: str = "tract_table",
) -> DualTableResultBundle:
    """Attach canonical + TRACT tables under stable names for one output bundle."""
    canonical = getattr(result, canonical_attr, None)
    tract = getattr(result, tract_attr, None)
    if not isinstance(canonical, pd.DataFrame):
        raise TypeError(
            f"Expected `{canonical_attr}` to be a pandas DataFrame; got {type(canonical).__name__}."
        )
    if not isinstance(tract, pd.DataFrame):
        raise TypeError(
            f"Expected `{tract_attr}` to be a pandas DataFrame; got {type(tract).__name__}."
        )
    return DualTableResultBundle(
        table=canonical,
        tract_table=tract,
        summary=getattr(result, "summary", None),
        request=getattr(result, "request", None),
    )


__all__ = ["DualTableResultBundle", "bundle_canonical_and_tract_tables"]
