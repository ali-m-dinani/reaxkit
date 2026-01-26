"""
Analyzer utilities for fort.76 (restraint monitor table).

Provides a simple API to extract user-requested columns
(e.g. iter, restraint 1 target, restraint 1 actual)
for plotting or downstream analysis.
"""

from __future__ import annotations

from typing import List, Sequence
import pandas as pd

from reaxkit.io.fort76_handler import Fort76Handler
from reaxkit.utils.alias import _resolve_alias, available_keys


# ---------- Internal helpers ----------

def _resolve_fort76_column(handler: Fort76Handler, requested: str) -> str:
    """
    Resolve a user-requested column name to a canonical fort.76 DataFrame column.

    Resolution order:
      1) Restraint columns: "restraint 1 target", "r1_actual", etc.
      2) Standard columns via alias.py: iter, E_res, E_pot
      3) Exact column name fallback
    """
    df = handler.dataframe()
    cols = list(df.columns)

    key = requested.strip().lower()

    # ---- 1) restraint columns (simple parsing, no regex) ----
    # accepted examples:
    #   "restraint 1 target value"
    #   "restraint 1 actual"
    #   "r1_target"
    #   "r2 actual"
    if "restraint" in key or key.startswith("r"):
        parts = key.replace("_", " ").split()

        # find index
        idx = None
        for p in parts:
            if p.isdigit():
                idx = int(p)
                break

        if idx is not None:
            if "target" in parts:
                col = f"r{idx}_target"
            elif "actual" in parts:
                col = f"r{idx}_actual"
            else:
                raise KeyError(
                    f"Restraint column must specify target or actual: '{requested}'"
                )

            if col not in cols:
                raise KeyError(
                    f"Resolved '{requested}' → '{col}', but column not found. "
                    f"Available restraint columns: {[c for c in cols if c.startswith('r')]}"
                )
            return col

    # ---- 2) standard columns via alias.py ----
    try:
        return _resolve_alias(df, requested)
    except KeyError:
        pass

    # ---- 3) exact match fallback ----
    if requested in cols:
        return requested

    raise KeyError(
        f"Could not resolve requested column '{requested}'. "
        f"Available columns / aliases: {available_keys(cols)}"
    )


# ---------- Public API ----------

def get_fort76_columns(
    handler: Fort76Handler,
    columns: Sequence[str],
    dropna_rows: bool = False,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Return a DataFrame with only the user-requested fort.76 columns.

    Example
    -------
    get_fort76_columns(
        handler,
        ["iter", "restraint 1 target value", "restraint 1 actual value"]
    )

    Parameters
    ----------
    handler : Fort76Handler
        Parsed fort.76 handler.
    columns : Sequence[str]
        User-requested column names (aliases allowed).
    dropna_rows : bool
        Drop rows where all selected non-iter columns are NaN.
    copy : bool
        Return a copy of the DataFrame.

    Returns
    -------
    pd.DataFrame
    """
    df = handler.dataframe()

    resolved: List[str] = [
        _resolve_fort76_column(handler, col) for col in columns
    ]

    out = df.loc[:, resolved]

    if dropna_rows:
        non_iter = [c for c in out.columns if c.lower() != "iter"]
        if non_iter:
            out = out.dropna(axis=0, how="all", subset=non_iter)

    return out.copy() if copy else out


def get_restraint_pair(
    handler: Fort76Handler,
    restraint_index: int,
    include_iter: bool = True,
) -> pd.DataFrame:
    """
    Convenience helper for a single restraint.

    Returns:
      ["iter", "rN_target", "rN_actual"]  (iter optional)
    """
    req: List[str] = []
    if include_iter:
        req.append("iter")

    req += [
        f"restraint {restraint_index} target value",
        f"restraint {restraint_index} actual value",
    ]

    return get_fort76_columns(handler, req)
