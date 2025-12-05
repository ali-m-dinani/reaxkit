"""analyzer for fort.73 file"""
from __future__ import annotations
from typing import List, Optional
import pandas as pd

from reaxkit.io.template_handler import TemplateHandler


def get_fort73_data(handler: TemplateHandler, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Return a DataFrame view of the fort.73 data.

    Parameters
    ----------
    handler : TemplateHandler
        Typically a Fort73Handler instance that has already parsed fort.73.
    columns : list[str], optional
        If provided, return only these columns (e.g., ["iter", "Ebond", "Evdw"]).
        If None, return the full DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing fort.73 data (with 'iter' as iteration column).
    """
    df = handler.dataframe().copy()

    if columns is not None:
        # Validate requested columns
        missing = set(columns) - set(df.columns)
        if missing:
            raise KeyError(f"Requested columns not found in fort.73 DataFrame: {missing}")
        df = df[columns]

    return df
