"""
Template analysis utilities.

This module provides example analysis functions that operate on data exposed by
``TemplateHandler`` (a minimal ReaxFF file handler). Use this template as a
starting point for new analysis modules.

Typical use cases include:

- extracting a small table of key columns for plotting/export
- computing derived metrics from a handler summary table
- building reusable helpers for workflows and CLI tasks
"""

from __future__ import annotations

from typing import List

import pandas as pd

from reaxkit.io.base_handler import FileHandler


def example_metric(handler: FileHandler) -> pd.DataFrame:
    """
    Extract iteration and energy as a minimal analysis table.

    Works on
    --------
    TemplateHandler — ``<filetype>``

    Parameters
    ----------
    handler : TemplateHandler
        Parsed handler instance exposing a summary ``dataframe()``.

    Returns
    -------
    pandas.DataFrame
        Table with columns: ``iteration``, ``energy``.

    Examples
    --------
    >>>
    """
    df = handler.dataframe()
    return df[["iteration", "energy"]].copy()


def record_series(handler: FileHandler, field: str) -> List[float]:
    """
    Extract a single column as a Python list across all rows.

    Works on
    --------
    TemplateHandler — ``<filetype>``

    Parameters
    ----------
    handler : TemplateHandler
        Parsed handler instance exposing a summary ``dataframe()``.
    field : str
        Column name to extract (e.g., ``"energy"``).

    Returns
    -------
    list[float]
        Values from ``handler.dataframe()[field]`` in row order.

    Examples
    --------
    >>>
    """
    return handler.dataframe()[field].tolist()
