"""
params (tunable-parameter list) analysis utilities.

This module provides helpers for working with ReaxFF ``params`` files via
``ParamsHandler``, and optionally interpreting each params entry as a pointer
into the corresponding ``ffield`` section via ``FFieldHandler``.

Typical use cases include:

- loading params tables with optional duplicate removal and sorting
- translating (ff_section, ff_section_line, ff_parameter) into an ffield parameter name/value
- attaching human-readable interaction labels (e.g., C-H, C-C-C) when available
"""


from __future__ import annotations

import pandas as pd
from typing import Dict, List, Tuple

from reaxkit.io.handlers.params_handler import ParamsHandler
from reaxkit.io.handlers.ffield_handler import FFieldHandler
from reaxkit.analysis.per_file.ffield_analyzer import interpret_one_section

def get_params_data(
    handler: ParamsHandler,
    *,
    sort_by: str | None = None,
    ascending: bool = True,
    drop_duplicate: bool = True
) -> pd.DataFrame:
    """
    Retrieve params entries as a DataFrame with optional sorting and de-duplication.

    Works on
    --------
    ParamsHandler — ``params`` / ``params.in``

    Parameters
    ----------
    handler : ParamsHandler
        Parsed params handler.
    sort_by : str, optional
        Column name to sort by (e.g. ``ff_section``, ``min_value``, ``max_value``).
        If None, rows are returned in file order.
    ascending : bool, default=True
        Sort order when ``sort_by`` is specified.
    drop_duplicate : bool, default=True
        If True, drop duplicate rows by ``(ff_section, ff_section_line, ff_parameter)``,
        keeping the first occurrence.

    Returns
    -------
    pandas.DataFrame
        Params table with columns such as:
        ``ff_section``, ``ff_section_line``, ``ff_parameter``,
        ``search_interval``, ``min_value``, ``max_value``, ``inline_comment``.

    Examples
    --------
    >>> from reaxkit.io.handlers.params_handler import ParamsHandler
    >>> from reaxkit.analysis.per_file.params_analyzer import get_params_data
    >>> h = ParamsHandler("params")
    >>> df = get_params_data(h, drop_duplicate=True)
    """
    df = handler.dataframe().copy()

    if drop_duplicate:
        df = df.drop_duplicates(
            subset=["ff_section", "ff_section_line", "ff_parameter"],
            keep="first"
        )

    if sort_by:
        if sort_by not in df.columns:
            raise ValueError(
                f"'sort_by' must be one of {list(df.columns)}, got {sort_by!r}"
            )
        df = df.sort_values(by=sort_by, ascending=ascending)

    return df

###############################################################################
# A “interpreter which translates a line like
# 3 49  1  1.0000   45.0   180.0
# bond data (because of ff_section = 3),
# line number 49 in that section,
# the first paramter in that line.
# These are all based on the ffield data.
###############################################################################

# ff_section number → canonical ffield section key + friendly name
_SECTION_NUM_MAP: Dict[int, Tuple[str, str]] = {
    1: (FFieldHandler.SECTION_GENERAL, "general"),
    2: (FFieldHandler.SECTION_ATOM, "atom"),
    3: (FFieldHandler.SECTION_BOND, "bond"),
    4: (FFieldHandler.SECTION_OFF_DIAGONAL, "off_diagonal"),
    5: (FFieldHandler.SECTION_ANGLE, "angle"),
    6: (FFieldHandler.SECTION_TORSION, "torsion"),
    7: (FFieldHandler.SECTION_HBOND, "hbond"),
}


# Which columns in each section are "index/identity" (NOT tunable parameters)
# Everything else (in original df column order) is treated as "parameter columns".
_SECTION_INDEX_COLS: Dict[str, List[str]] = {
    FFieldHandler.SECTION_GENERAL: [],
    FFieldHandler.SECTION_ATOM: ["symbol"],            # adjust if your atom df has other identity cols
    FFieldHandler.SECTION_BOND: ["i", "j"],
    FFieldHandler.SECTION_OFF_DIAGONAL: ["i", "j"],
    FFieldHandler.SECTION_ANGLE: ["i", "j", "k"],
    FFieldHandler.SECTION_TORSION: ["i", "j", "k", "l"],
    FFieldHandler.SECTION_HBOND: ["i", "j", "k"],
}


def _param_columns_for_section(sec_df: pd.DataFrame, section_key: str) -> List[str]:
    """
    Return the ordered list of parameter columns for a given ffield section df.
    We treat all non-index columns (based on _SECTION_INDEX_COLS) as tunable parameters.
    """
    idx_cols = set(_SECTION_INDEX_COLS.get(section_key, []))
    return [c for c in sec_df.columns if c not in idx_cols]


def interpret_params(
    params_handler: ParamsHandler,
    ffield_handler: FFieldHandler,
    *,
    add_term: bool = True,
    sep: str = "-",
) -> pd.DataFrame:
    """
    Interpret each params row as a pointer into the corresponding ffield section.

    Each params entry points to an ffield value using:

    - ``ff_section``: section number (1..7 → general, atom, bond, off-diagonal, angle, torsion, hbond)
    - ``ff_section_line``: 1-based row number within that ffield section
    - ``ff_parameter``: 1-based index of the tunable parameter within that row

    Works on
    --------
    ParamsHandler + FFieldHandler — ``params`` + ``ffield``

    Parameters
    ----------
    params_handler : ParamsHandler
        Parsed params handler.
    ffield_handler : FFieldHandler
        Parsed ffield handler.
    add_term : bool, default=True
        If True, include a human-readable interaction label (``term``) for
        multi-body sections when available (e.g. ``C-H``, ``C-C-C``).
    sep : str, default="-"
        Separator used for building ``term`` labels.

    Returns
    -------
    pandas.DataFrame
        Interpreted params table including the original params fields plus:
        - ``ffield_section_key`` and ``ffield_section_name``
        - ``ffield_row_index`` (0-based row index)
        - ``ffield_param_name`` (parameter column name in ffield)
        - ``ffield_value`` (current value from ffield)
        - ``term`` (optional interaction label)

    Examples
    --------
    >>> from reaxkit.io.handlers.params_handler import ParamsHandler
    >>> from reaxkit.io.handlers.ffield_handler import FFieldHandler
    >>> from reaxkit.analysis.per_file.params_analyzer import interpret_params
    >>> p = ParamsHandler("params")
    >>> f = FFieldHandler("ffield")
    >>> df = interpret_params(p, f, add_term=True)
    """
    p = params_handler.dataframe().copy()

    out_rows: List[Dict[str, object]] = []

    # Cache section dfs (optionally interpreted w/ symbols)
    sec_cache: Dict[str, pd.DataFrame] = {}

    for r in p.itertuples(index=False):
        sec_num = int(getattr(r, "ff_section"))
        line_1b = int(getattr(r, "ff_section_line"))
        par_1b = int(getattr(r, "ff_parameter"))

        if sec_num not in _SECTION_NUM_MAP:
            raise ValueError(f"Unknown ff_section={sec_num}. Expected 1..7.")

        section_key, section_name = _SECTION_NUM_MAP[sec_num]

        # Load section df (and optionally add term)
        if section_key not in sec_cache:
            base_df = ffield_handler.section_df(section_key).copy()
            if add_term and section_key in {
                FFieldHandler.SECTION_BOND,
                FFieldHandler.SECTION_OFF_DIAGONAL,
                FFieldHandler.SECTION_ANGLE,
                FFieldHandler.SECTION_TORSION,
                FFieldHandler.SECTION_HBOND,
            }:
                # adds i_symbol/j_symbol/... and 'term'
                base_df = interpret_one_section(ffield_handler, section=section_name, sep=sep)
            sec_cache[section_key] = base_df

        sec_df = sec_cache[section_key]

        row_idx = line_1b - 1
        if row_idx < 0 or row_idx >= len(sec_df):
            raise IndexError(
                f"params points to {section_name} line {line_1b}, "
                f"but section has {len(sec_df)} rows."
            )

        param_cols = _param_columns_for_section(sec_df, section_key)

        # Safety: allow params that point to "term" or "*_symbol" columns only if user wants that
        # By default, those are NOT included because they're not in _SECTION_INDEX_COLS,
        # so we exclude them explicitly here.
        param_cols = [c for c in param_cols if not (c.endswith("_symbol") or c == "term")]

        par_idx = par_1b - 1
        if par_idx < 0 or par_idx >= len(param_cols):
            raise IndexError(
                f"params points to {section_name} parameter {par_1b}, "
                f"but only {len(param_cols)} parameter columns exist: {param_cols}"
            )

        param_name = param_cols[par_idx]
        row = sec_df.iloc[row_idx]
        val = row[param_name]

        term = row.get("term") if add_term else None

        out_rows.append(
            {
                # original params fields
                "ff_section": sec_num,
                "ff_section_line": line_1b,
                "ff_parameter": par_1b,
                "search_interval": getattr(r, "search_interval"),
                "min_value": getattr(r, "min_value"),
                "max_value": getattr(r, "max_value"),
                "inline_comment": getattr(r, "inline_comment"),

                # interpreted fields
                "ffield_section_key": section_key,
                "ffield_section_name": section_name,
                "ffield_row_index": row_idx,        # 0-based
                "ffield_param_name": param_name,
                "ffield_value": val,
                "term": term,
            }
        )

    return pd.DataFrame(out_rows)
