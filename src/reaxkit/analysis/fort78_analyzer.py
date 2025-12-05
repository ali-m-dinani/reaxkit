"""analyzer for fort.78 file"""

from __future__ import annotations
from typing import Sequence
import pandas as pd

from reaxkit.io.fort78_handler import Fort78Handler
from reaxkit.utils.alias import resolve_alias_from_columns, _DEFAULT_ALIAS_MAP
from reaxkit.io.control_handler import ControlHandler
from reaxkit.analysis.control_analyzer import get_control

def get_iter_vs_fort78_data(handler: Fort78Handler, variables: str | Sequence[str]) -> pd.DataFrame:
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

# ----------------------------------------------------------------------------------------
# matching electric fields data to iout2, which is used to get plots of summary,
# xmolout, polarization, or etc. along with electric field profile
# ----------------------------------------------------------------------------------------

def match_electric_field_to_iout2(
    f78: Fort78Handler,
    ctrl: ControlHandler,
    target_iters: Sequence[int],
    field_var: str = "E_field_z",
) -> pd.Series:
    """
    Match electric field values from fort.78 to a set of target iterations.

    Uses a piecewise-constant rule:
      - For each target iter, pick the last fort.78 iter <= target_iter
      - If target_iter == 0, return 0.0 by convention.

    Example:
      fort.78 iters: 0, 80, 160
      target iter:   100
      → uses value at iter=80

    Parameters
    ----------
    f78 : Fort78Handler
    ctrl : ControlHandler
        Used to access iout2 if needed for sanity checks.
    target_iters : sequence of int
    field_var : str
        Electric field column to use (e.g. 'E_field_z', 'field_z', etc.).

    Returns
    -------
    pd.Series
        index: target_iters
        values: matched field_var values (same units as fort.78, typically V/Å)
    """
    # Access iout2 from control (not strictly needed for the matching logic,
    # but useful for consistency/sanity checks with xmolout/summary output rate).
    iout2 = control_get(ctrl, "iout2", section="md", default=1)
    _ = iout2  # currently not used explicitly, but kept for future use/logging

    df_E = get_iter_vs_fort78_data(f78, variables=field_var)  # columns: ['iter', field_var]
    if df_E.empty:
        raise ValueError("fort.78 has no usable data.")

    df_E = df_E.sort_values("iter").reset_index(drop=True)
    mapping = pd.Series(df_E[field_var].values, index=df_E["iter"].values)
    sorted_iters = list(mapping.index)

    out_vals = []
    for it in target_iters:
        it = int(it)
        if it == 0:
            out_vals.append(0.0)
            continue
        # all fort.78 iters <= this iter
        valid = [f for f in sorted_iters if f <= it]
        if not valid:
            raise ValueError(f"No fort.78 iteration ≤ {it}")
        out_vals.append(float(mapping[max(valid)]))

    return pd.Series(out_vals, index=list(target_iters), name=field_var)