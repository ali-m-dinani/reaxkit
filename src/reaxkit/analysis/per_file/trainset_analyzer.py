"""
trainset (ReaxFF training-set) analysis utilities.

This module provides helper functions for extracting metadata and
human-readable annotations from ReaxFF ``trainset`` files via
``TrainsetHandler``.

Typical use cases include:

- listing unique group comments defined across training sections
- auditing how training targets are grouped and documented
- preparing summaries of training-set structure for reporting
"""


from __future__ import annotations

from typing import Any
import pandas as pd

from reaxkit.io.handlers.trainset_handler import TrainsetHandler


def get_trainset_group_comments(handler: TrainsetHandler, *, sort: bool = False) -> pd.DataFrame:
    """
    Collect unique group comments from a ReaxFF training-set file.

    Each group comment is returned together with the training section
    it belongs to.

    Works on
    --------
    TrainsetHandler — ``trainset``

    Parameters
    ----------
    handler : TrainsetHandler
        Parsed trainset handler with metadata and section tables.
    sort : bool, default=False
        If True, sort the result by ``section`` and ``group_comment``.
        If False, preserve the original appearance order.

    Returns
    -------
    pandas.DataFrame
        Table with columns:
        ``section`` — training section name
        ``group_comment`` — unique group annotation text

    Examples
    --------
    >>> from reaxkit.io.handlers.trainset_handler import TrainsetHandler
    >>> from reaxkit.analysis.per_file.trainset_analyzer import get_trainset_group_comments
    >>> h = TrainsetHandler("trainset")
    >>> df = get_trainset_group_comments(h, sort=True)
    """
    meta: dict[str, Any] = handler.metadata()
    tables: dict[str, pd.DataFrame] = meta.get("tables", {})

    rows: list[dict[str, str]] = []

    for section_name, df in tables.items():
        if "group_comment" not in df.columns:
            continue

        # keep only non-empty comments
        series = df["group_comment"].astype(str).str.strip()
        series = series[series != ""]

        for gc in series.unique():
            rows.append(
                {
                    "section": section_name.lower(),  # or keep as-is
                    "group_comment": gc,
                }
            )

    result = pd.DataFrame(rows).drop_duplicates()

    # Apply sort only if requested
    if sort and not result.empty:
        result = result.sort_values(["section", "group_comment"], ignore_index=True)

    return result
