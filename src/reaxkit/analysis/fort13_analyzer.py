"""analyzer for fort.13 file"""
from __future__ import annotations
from typing import List, Optional
import pandas as pd

from reaxkit.io.fort13_handler import Fort13Handler


def get_errors(handler: Fort13Handler, epochs: Optional[List[int]] = None) -> pd.DataFrame:
    """Return total errors for all or selected epochs (lines) in fort.13.

    Parameters
    ----------
    handler : Fort13Handler
        The handler instance that has parsed fort.13 data.
    epochs : list[int], optional
        List of specific epoch numbers (epochs) to extract.
        If None, all epochs are returned.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'epoch' and 'total_ff_error' columns.
    """
    df = handler.dataframe()

    if epochs is not None:
        df = df[df["epoch"].isin(epochs)].reset_index(drop=True)

    return df[["epoch", "total_ff_error"]].copy()


def error_series(handler: Fort13Handler) -> List[float]:
    """Return the total_ff_error values across all epochs as a list.
    """
    return handler.dataframe()["total_ff_error"].tolist()
