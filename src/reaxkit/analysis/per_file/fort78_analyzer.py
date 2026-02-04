"""
fort.78 (external field schedule) analysis utilities.

This module provides helpers for extracting and aligning electric-field
schedule data written by ReaxFF into ``fort.78`` files via ``Fort78Handler``.

Typical use cases include:

- extracting electric-field components versus iteration
- matching field values to other outputs sampled at ``iout2`` frequency
- preparing aligned field profiles for plotting with xmolout or summaries
"""


from __future__ import annotations
from typing import Sequence
import pandas as pd

from reaxkit.io.handlers.fort78_handler import Fort78Handler
from reaxkit.utils.alias import resolve_alias_from_columns, load_default_alias_map
from reaxkit.io.handlers.control_handler import ControlHandler
from reaxkit.analysis.per_file.control_analyzer import get_control_data

def get_fort78_data(handler: Fort78Handler, variables: str | Sequence[str]) -> pd.DataFrame:
    """Extract iteration versus one or more variables from a ``fort.78`` file.

    Works on
    --------
    Fort78Handler — ``fort.78``

    Parameters
    ----------
    handler : Fort78Handler
        Parsed ``fort.78`` handler.
    variables : str or sequence of str
        Field variable name(s) to extract. Aliases are supported
        (e.g. ``E_field_z``, ``field_z``). Output columns are named
        exactly as requested.

    Returns
    -------
    pandas.DataFrame
        Table with columns ``["iter", <variables...>]`` where
        ``<variables>`` correspond to the requested names.

    Examples
    --------
    >>> from reaxkit.io.handlers.fort78_handler import Fort78Handler
    >>> from reaxkit.analysis.per_file.fort78_analyzer import get_fort78_data
    >>> h = Fort78Handler("fort.78")
    >>> df = get_fort78_data(h, variables=["E_field_z", "E_field_x"])
    """
    df = handler.dataframe()
    cols = list(df.columns)

    if isinstance(variables, str):
        variables = [variables]

    # Resolve iter, then always expose as 'iter'
    iter_hit = resolve_alias_from_columns(cols, "iter", load_default_alias_map())
    if iter_hit is None:
        raise KeyError(f"'iter' column not found. Available: {cols}")

    out = df[[iter_hit]].copy()
    if iter_hit != "iter":
        out = out.rename(columns={iter_hit: "iter"})

    # Resolve and rename each requested var to the *requested name*
    for var in variables:
        hit = resolve_alias_from_columns(cols, var, load_default_alias_map())
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
    Match electric-field values from ``fort.78`` to target iterations.

    The matching follows a piecewise-constant rule:
    for each target iteration, the last available fort.78 value with
    ``iter <= target_iter`` is used. By convention, ``iter = 0`` maps to 0.0.

    Works on
    --------
    Fort78Handler + ControlHandler — ``fort.78`` + ``control``

    Parameters
    ----------
    f78 : Fort78Handler
        Parsed ``fort.78`` handler providing the field schedule.
    ctrl : ControlHandler
        Parsed control handler (used to access ``iout2`` for consistency).
    target_iters : sequence of int
        Iteration numbers to which the electric field should be matched.
    field_var : str, default="E_field_z"
        Electric-field component to use (aliases supported).

    Returns
    -------
    pandas.Series
        Series indexed by ``target_iters`` containing matched electric-field
        values (same units as ``fort.78``, typically V/Å).

    Examples
    --------
    >>> field = match_electric_field_to_iout2(
    ...     f78, ctrl, target_iters=[0, 80, 160], field_var="E_field_z"
    ... )
    """
    # Access iout2 from control (not strictly needed for the matching logic,
    # but useful for consistency/sanity checks with xmolout/summary output rate).
    iout2 = get_control_data(ctrl, "iout2", section="md", default=1)
    _ = iout2  # currently not used explicitly, but kept for future use/logging

    df_E = get_fort78_data(f78, variables=field_var)  # columns: ['iter', field_var]
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