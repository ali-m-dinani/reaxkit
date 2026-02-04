"""
fort.13 (training error) analysis utilities.

This module provides helper functions for extracting force-field training
error values from a parsed ReaxFF ``fort.13`` file via ``Fort13Handler``.

Typical use cases include:

- retrieving total force-field error versus epoch
- extracting the error series for quick plotting or post-processing
"""


from __future__ import annotations
from typing import List, Optional
import pandas as pd

from reaxkit.io.handlers.fort13_handler import Fort13Handler


def get_fort13_data(handler: Fort13Handler, epochs: Optional[List[int]] = None) -> pd.DataFrame:
    """Extract total force-field error for all or selected epochs from ``fort.13``.

    Works on
    --------
    Fort13Handler — ``fort.13``

    Parameters
    ----------
    handler : Fort13Handler
        Parsed ``fort.13`` handler.
    epochs : list[int], optional
        Epoch numbers to include. If None, all epochs are returned.

    Returns
    -------
    pandas.DataFrame
        Error table with columns: ``epoch`` and ``total_ff_error``.

    Examples
    --------
    >>> from reaxkit.io.handlers.fort13_handler import Fort13Handler
    >>> from reaxkit.analysis.per_file.fort13_analyzer import get_fort13_data
    >>> h = Fort13Handler("fort.13")
    >>> df = get_fort13_data(h, epochs=[1, 10, 50])
    """
    df = handler.dataframe()

    if epochs is not None:
        df = df[df["epoch"].isin(epochs)].reset_index(drop=True)

    return df[["epoch", "total_ff_error"]].copy()


def _error_series_across_epochs(handler: Fort13Handler) -> List[float]:
    """Return the total force-field error values across all epochs as a list.

    Works on
    --------
    Fort13Handler — ``fort.13``

    Parameters
    ----------
    handler : Fort13Handler
        Parsed ``fort.13`` handler.

    Returns
    -------
    list[float]
        Total force-field error values in epoch order.

    Examples
    --------
    >>> from reaxkit.io.handlers.fort13_handler import Fort13Handler
    >>> from reaxkit.analysis.per_file.fort13_analyzer import _error_series_across_epochs
    >>> h = Fort13Handler("fort.13")
    >>> errs = _error_series_across_epochs(h)
    """
    return handler.dataframe()["total_ff_error"].tolist()
