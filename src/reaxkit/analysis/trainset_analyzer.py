from __future__ import annotations

from typing import Any
import pandas as pd

from reaxkit.io.trainset_handler import TrainsetHandler


def trainset_group_comments(handler: TrainsetHandler, *, sort: bool = False) -> pd.DataFrame:
    """
    Collect all UNIQUE group comments from a trainset file,
    along with the section they belong to.

    Parameters
    ----------
    handler : TrainsetHandler
        Parsed trainset handler with metadata and tables.

    sort : bool, default=True
        If True, sort the result by 'section' and 'group_comment'.
        If False, preserve original appearance order.

    Returns
    -------
    DataFrame with columns:
        - section
        - group_comment
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
