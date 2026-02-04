"""
fort.74 (thermodynamic summary) analysis utilities.

This module provides helpers for accessing and sorting thermodynamic
summary data produced by ReaxFF in ``fort.74`` files via ``Fort74Handler``.

Typical use cases include:

- retrieving formation energies, volumes, or densities
- sorting structures by energy or iteration
- exporting fort.74 data for post-processing or plotting
"""


from __future__ import annotations
import pandas as pd
from typing import Optional

from reaxkit.io.handlers.fort74_handler import Fort74Handler


def get_fort74_data(
    handler: Fort74Handler,
    *,
    sort: Optional[str] = None,
    ascending: bool = True
) -> pd.DataFrame:
    """
    Retrieve thermodynamic summary data from a ``fort.74`` file.

    Works on
    --------
    Fort74Handler — ``fort.74``

    Parameters
    ----------
    handler : Fort74Handler
        Parsed ``fort.74`` handler.
    sort : str, optional
        Column name or alias to sort by (e.g. ``Emin``, ``Hf``, ``iter``).
        If None, rows are returned in file order.
    ascending : bool, default=True
        Sort order when ``sort`` is specified.

    Returns
    -------
    pandas.DataFrame
        Table containing thermodynamic and structural summary quantities
        from ``fort.74``.

    Examples
    --------
    >>> from reaxkit.io.handlers.fort74_handler import Fort74Handler
    >>> from reaxkit.analysis.per_file.fort74_analyzer import get_fort74_data
    >>> h = Fort74Handler("fort.74")
    >>> df = get_fort74_data(h, sort="Emin")
    """
    df = handler.dataframe().copy()

    # No sorting → return as-is
    if not sort:
        return df

    # Sort and return
    return df.sort_values(by=sort, ascending=ascending).reset_index(drop=True)
