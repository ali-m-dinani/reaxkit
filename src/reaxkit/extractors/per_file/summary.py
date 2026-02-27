"""Structured summary.txt extraction helpers."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import pandas as pd

from reaxkit.core.alias import _resolve_alias, available_keys, normalize_choice
from reaxkit.engine.reaxff.io.summary_handler import SummaryHandler


def extract_summary_data(
    handler: SummaryHandler,
    feature: str,
    frames: Optional[Union[slice, Sequence[int]]] = None,
) -> pd.Series:
    """Extract a single summary quantity from summary.txt as a pandas Series."""
    canonical = normalize_choice(feature)

    try:
        col = _resolve_alias(handler, canonical)
    except KeyError:
        cols_lower = {c.lower(): c for c in handler.dataframe().columns}
        direct = cols_lower.get(feature.strip().lower())
        if direct is None:
            raise KeyError(
                f"Column '{feature}' not found. "
                f"Try one of: {available_keys(handler.dataframe().columns)}"
            )
        col = direct

    s = handler.dataframe()[col]
    if frames is None:
        return s.copy()
    if isinstance(frames, slice):
        return s.iloc[frames].copy()
    return s.iloc[list(frames)].copy()


__all__ = ["extract_summary_data"]
