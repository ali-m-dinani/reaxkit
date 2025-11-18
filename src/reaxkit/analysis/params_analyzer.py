from __future__ import annotations
import pandas as pd
from reaxkit.io.params_handler import ParamsHandler


def get_params(
    handler: ParamsHandler,
    *,
    sort_by: str | None = None,
    ascending: bool = True,
    drop_duplicate: bool = True
) -> pd.DataFrame:
    """
    Return the full parameters DataFrame, with options for sorting
    and removing duplicate parameter definitions.

    Parameters
    ----------
    handler : ParamsHandler
        Parsed handler for params.in or similar file.

    sort_by : str, optional
        Column name to sort by (e.g., 'ff_section', 'min_value', etc.)

    ascending : bool, optional (default=True)
        Sort order.

    drop_duplicate : bool, optional (default=False)
        If True, drops rows where (ff_section, ff_section_line, ff_parameter)
        appear more than once, keeping only the first occurrence.

    Returns
    -------
    pd.DataFrame
    """
    df = handler.dataframe().copy()

    if drop_duplicate:
        df = df.drop_duplicates(
            subset=["ff_section", "ff_section_line", "ff_parameter"],
            keep="first"
        )

    if sort_by:
        if sort_by not in df.columns:
            raise ValueError(
                f"'sort_by' must be one of {list(df.columns)}, got {sort_by!r}"
            )
        df = df.sort_values(by=sort_by, ascending=ascending)

    return df
