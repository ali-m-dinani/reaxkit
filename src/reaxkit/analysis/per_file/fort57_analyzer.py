"""
fort.57 (geometry / convergence) analysis utilities.

This module provides helpers for extracting selected convergence and
geometry-related quantities from a parsed ReaxFF ``fort.57`` file
via ``Fort57Handler``.

Typical use cases include:

- selecting canonical fort.57 columns with alias support
- exporting convergence metrics such as RMSG or potential energy
- attaching a geometry descriptor for multi-structure workflows
"""


from __future__ import annotations

from typing import Sequence

import pandas as pd

from reaxkit.io.handlers.fort57_handler import Fort57Handler
from reaxkit.utils.alias import normalize_choice, resolve_alias_from_columns


_F57_CANONICAL = ("iter", "E_pot", "T", "T_set", "RMSG", "nfc")


def get_fort57_data(
    *,
    fort57_handler: Fort57Handler,
    cols: Sequence[str] | None = None,
    include_geo_descriptor: bool = False,
) -> pd.DataFrame:
    """
    Extract selected columns from a ``fort.57`` file as a DataFrame.

    Works on
    --------
    Fort57Handler — ``fort.57``

    Parameters
    ----------
    fort57_handler : Fort57Handler
        Parsed ``fort.57`` handler.
    cols : sequence of str, optional
        Columns to extract using canonical names or aliases
        (e.g. ``iter``, ``E_pot``, ``T``, ``T_set``, ``RMSG``, ``nfc``).
        If None, all available columns are returned.
    include_geo_descriptor : bool, default=False
        If True, prepend a ``geo_descriptor`` column identifying
        the geometry associated with this data.

    Returns
    -------
    pandas.DataFrame
        Table containing the requested columns. Column names are
        normalized to canonical fort.57 keys.

    Examples
    --------
    >>> from reaxkit.io.handlers.fort57_handler import Fort57Handler
    >>> from reaxkit.analysis.per_file.fort57_analyzer import get_fort57_data
    >>> h = Fort57Handler("fort.57")
    >>> df = get_fort57_data(fort57_handler=h, cols=["iter", "RMSG"])
    """
    df = fort57_handler.dataframe()

    # default: all canonical columns (if present)
    if cols is None or len(cols) == 0:
        out = df.copy()
    else:
        # 1) Normalize user requests to canonical keys (via alias.py)
        wanted_canon = [normalize_choice(c, domain="fort57.md") for c in cols]

        # 2) Resolve those canonical keys to actual dataframe columns
        #    (usually identical, but this keeps it robust)
        resolved_cols: list[str] = []
        available = list(df.columns)

        for canon in wanted_canon:
            if canon not in _F57_CANONICAL:
                raise ValueError(
                    f"Unknown fort.57 column '{canon}'. "
                    f"Allowed: {', '.join(_F57_CANONICAL)}"
                )

            actual = resolve_alias_from_columns(available, canon)
            if actual is None:
                raise KeyError(
                    f"Column '{canon}' not found (and no alias matched). "
                    f"Available: {', '.join(available)}"
                )
            resolved_cols.append(actual)

        out = df.loc[:, resolved_cols].copy()

        # Optional: rename resolved columns back to canonical names
        # so downstream code always sees iter/E_pot/T/T_set/RMSG/nfc
        rename_map = dict(zip(resolved_cols, wanted_canon))
        out = out.rename(columns=rename_map)

    if include_geo_descriptor:
        out.insert(0, "geo_descriptor", fort57_handler.geo_descriptor)

    return out
