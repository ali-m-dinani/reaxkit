"""analyzer for summary.txt file"""

from __future__ import annotations
from typing import Optional, Sequence, Union
import pandas as pd

from reaxkit.io.summary_handler import SummaryHandler
from reaxkit.utils.alias import _resolve_alias, available_keys, normalize_choice

__all__ = ["get_summary"]


def get_summary(
    handler: SummaryHandler,
    feature: str,
    frames: Optional[Union[slice, Sequence[int]]] = None,
) -> pd.Series:
    """Return a column from the parsed summary as a Series, and used to get summary.txt data.

    Accepts canonical names (e.g., 'E_pot') or legacy aliases (e.g., 'Epot(kcal/mol)').
    """
    # Map legacy -> canonical (e.g., "Epot(kcal/mol)" -> "E_pot")
    canonical = normalize_choice(feature)

    # Resolve against dataframe columns
    try:
        col = _resolve_alias(handler, canonical)
    except KeyError:
        # Fallback: case-insensitive direct match
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
