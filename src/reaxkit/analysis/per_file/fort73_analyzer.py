"""
Energy output analysis utilities (fort.73, energylog, fort.58).

This module provides a lightweight interface for accessing energy-related
output written by ReaxFF during MD, MM, or optimization runs. It supports
``fort.73``, ``energylog``, and ``fort.58`` files via a common handler interface.

Typical use cases include:

- extracting selected energy terms versus iteration
- exporting energy components for plotting or post-processing
- working uniformly with fort.73, energylog, and fort.58 outputs
"""


from __future__ import annotations
from typing import List, Optional
import pandas as pd

from reaxkit.io.base_handler import BaseHandler


def get_fort73_data(handler: BaseHandler, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Extract energy data from a ReaxFF energy output file as a DataFrame.

    Works on
    --------
    Fort73Handler / EnergylogHandler / Fort58Handler — ``fort.73``, ``energylog``, ``fort.58``

    Parameters
    ----------
    handler : TemplateHandler
        Parsed handler instance for ``fort.73``, ``energylog``, or ``fort.58``.
    columns : list[str], optional
        Energy columns to extract (e.g. ``["iter", "Ebond", "Evdw"]``).
        If None, all available columns are returned.

    Returns
    -------
    pandas.DataFrame
        Energy table indexed by iteration, containing the requested
        energy components.

    Examples
    --------
    >>> from reaxkit.io.handlers.fort73_handler import Fort73Handler
    >>> from reaxkit.analysis.per_file.fort73_analyzer import get_fort73_data
    >>> h = Fort73Handler("fort.73")
    >>> df = get_fort73_data(h, columns=["iter", "Ebond", "Evdw"])
    """
    df = handler.dataframe().copy()

    if columns is not None:
        # Validate requested columns
        missing = set(columns) - set(df.columns)
        if missing:
            raise KeyError(f"Requested columns not found in fort.73 DataFrame: {missing}")
        df = df[columns]

    return df
