"""Structured fort.78 extraction helpers."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from reaxkit.core.alias import load_default_alias_map, resolve_alias_from_columns
from reaxkit.engine.reaxff.io.fort78_handler import Fort78Handler


def extract_fort78_data(handler: Fort78Handler, variables: str | Sequence[str]) -> pd.DataFrame:
    """Extract iteration versus one or more variables from a fort.78 file."""
    df = handler.dataframe()
    cols = list(df.columns)

    if isinstance(variables, str):
        variables = [variables]

    iter_hit = resolve_alias_from_columns(cols, "iter", load_default_alias_map())
    if iter_hit is None:
        raise KeyError(f"'iter' column not found. Available: {cols}")

    out = df[[iter_hit]].copy()
    if iter_hit != "iter":
        out = out.rename(columns={iter_hit: "iter"})

    for var in variables:
        hit = resolve_alias_from_columns(cols, var, load_default_alias_map())
        if hit is None:
            raise KeyError(f"Variable '{var}' not found. Available: {cols}")
        out[var] = pd.to_numeric(df[hit], errors="coerce")

    out["iter"] = pd.to_numeric(out["iter"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["iter"])
    out["iter"] = out["iter"].astype(int)
    return out


__all__ = ["extract_fort78_data"]
