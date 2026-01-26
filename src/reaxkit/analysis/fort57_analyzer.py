# reaxkit/analysis/fort57_analyzer.py
from __future__ import annotations

from typing import Sequence

import pandas as pd

from reaxkit.io.fort57_handler import Fort57Handler
from reaxkit.utils.alias import normalize_choice, resolve_alias_from_columns


_F57_CANONICAL = ("iter", "E_pot", "T", "T_set", "RMSG", "nfc")


def fort57_get(
    *,
    fort57_handler: Fort57Handler,
    cols: Sequence[str] | None = None,
    include_geo_descriptor: bool = False,
) -> pd.DataFrame:
    """
    Return selected columns from fort.57 as a DataFrame.

    Examples
    --------
    df = fort57_get(fort57_handler=h, cols=["iter", "RMSG"])
    """
    df = fort57_handler.dataframe()

    # default: all canonical columns (if present)
    if cols is None or len(cols) == 0:
        out = df.copy()
    else:
        # 1) Normalize user requests to canonical keys (via alias.py)
        wanted_canon = [normalize_choice(c, domain="fort57") for c in cols]

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
