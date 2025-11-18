from __future__ import annotations

from typing import Any
import pandas as pd

from reaxkit.io.template_handler import TemplateHandler


def trainset_group_comments(handler: TemplateHandler) -> pd.DataFrame:
    """
    Collect all UNIQUE group comments from a trainset file,
    along with the section they belong to.

    Returns a DataFrame with columns:
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

    # optional: sort by section then group_comment
    if not result.empty:
        result = result.sort_values(["section", "group_comment"], ignore_index=True)

    return result
