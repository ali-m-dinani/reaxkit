"""analyzer for fort.78 file"""
from __future__ import annotations
from typing import Sequence
import pandas as pd

from reaxkit.io.fort78_handler import Fort78Handler
from reaxkit.utils.alias_utils import resolve_alias_from_columns, _DEFAULT_ALIAS_MAP

def get_iter_vs(handler: Fort78Handler, variables: str | Sequence[str]) -> pd.DataFrame:
    """Return iter vs one or more user-specified variables to get fort.78 data.
    The output will rename columns to the *requested alias(es)* so downstream
    code can safely look them up by what the user typed.

    Returns
    -------
    pd.DataFrame
        Columns: ['iter', <variables...>]  (variables use the *requested* names)
    """
    df = handler.dataframe()
    cols = list(df.columns)

    if isinstance(variables, str):
        variables = [variables]

    # Resolve iter, then always expose as 'iter'
    iter_hit = resolve_alias_from_columns(cols, "iter", _DEFAULT_ALIAS_MAP)
    if iter_hit is None:
        raise KeyError(f"'iter' column not found. Available: {cols}")

    out = df[[iter_hit]].copy()
    if iter_hit != "iter":
        out = out.rename(columns={iter_hit: "iter"})

    # Resolve and rename each requested var to the *requested name*
    for var in variables:
        hit = resolve_alias_from_columns(cols, var, _DEFAULT_ALIAS_MAP)
        if hit is None:
            raise KeyError(f"Variable '{var}' not found. Available: {cols}")
        # attach numeric-ified column using the requested alias name
        ser = pd.to_numeric(df[hit], errors="coerce")
        out[var] = ser

    # Ensure iter is numeric/int
    out["iter"] = pd.to_numeric(out["iter"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["iter"])
    out["iter"] = out["iter"].astype(int)

    return out
