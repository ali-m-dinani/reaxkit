"""analyzer for fort.74 file"""

from __future__ import annotations
import pandas as pd
from typing import Optional

from reaxkit.io.fort74_handler import Fort74Handler


def get_fort74(
    handler: Fort74Handler,
    *,
    sort: Optional[str] = None,
    ascending: bool = True
) -> pd.DataFrame:
    """
    Return the parsed fort.74 dataframe, with optional sorting.

    Parameters
    ----------
    handler : Fort74Handler
        Parsed file handler
    sort : str, optional
        A key/alias to sort by (e.g., 'Emin', 'Hf', 'iter')
    ascending : bool
        Sort order (default: True)

    Returns
    -------
    pd.DataFrame
        Resulting dataframe
    """
    df = handler.dataframe().copy()

    # No sorting → return as-is
    if not sort:
        return df

    # Sort and return
    return df.sort_values(by=sort, ascending=ascending).reset_index(drop=True)
