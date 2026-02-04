"""
summary.txt analysis utilities.

This module provides helper functions for accessing scalar, per-frame
summary quantities written by ReaxFF into ``summary.txt`` files via
``SummaryHandler``.

Typical use cases include:

- extracting a single summary quantity (e.g. potential energy) as a time series
- selecting subsets of frames for post-processing or plotting
- working with canonical column names and legacy aliases transparently
"""


from __future__ import annotations
from typing import Optional, Sequence, Union
import pandas as pd

from reaxkit.io.handlers.summary_handler import SummaryHandler
from reaxkit.utils.alias import _resolve_alias, available_keys, normalize_choice

__all__ = ["get_summary_data"]


def get_summary_data(
    handler: SummaryHandler,
    feature: str,
    frames: Optional[Union[slice, Sequence[int]]] = None,
) -> pd.Series:
    """Extract a single summary quantity from ``summary.txt`` as a pandas Series.

    Canonical column names (e.g. ``E_pot``) and legacy aliases
    (e.g. ``Epot(kcal/mol)``) are both supported.

    Works on
    --------
    SummaryHandler — ``summary.txt``

    Parameters
    ----------
    handler : SummaryHandler
        Parsed ``summary.txt`` handler.
    feature : str
        Name or alias of the summary quantity to extract.
    frames : slice or sequence of int, optional
        Frame indices to include. If None, all frames are returned.

    Returns
    -------
    pandas.Series
        Series containing the requested summary quantity, indexed by frame.

    Examples
    --------
    >>> from reaxkit.io.handlers.summary_handler import SummaryHandler
    >>> from reaxkit.analysis.per_file.summary_analyzer import get_summary_data
    >>> h = SummaryHandler("summary.txt")
    >>> epot = get_summary_data(h, "E_pot")
    >>> epot_head = get_summary_data(h, "Epot(kcal/mol)", frames=slice(0, 10))
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
