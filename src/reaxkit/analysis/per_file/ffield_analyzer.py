"""
ReaxFF force-field (ffield) analysis utilities.

This module provides helper functions for extracting and interpreting
sections of a ReaxFF ``ffield`` file via ``FFieldHandler``.

Typical use cases include:

- retrieving raw parameter tables for specific ffield sections
- interpreting numeric atom indices (e.g. 1–1) into chemical symbols (e.g. C–C)
- generating human-readable interaction labels for bonds, angles, torsions, and H-bonds
"""


from __future__ import annotations
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd

from reaxkit.io.base_handler import BaseHandler
from reaxkit.io.handlers.ffield_handler import FFieldHandler


# Map user-friendly names to canonical section keys
_SECTION_ALIASES: Dict[str, str] = {
    "general": FFieldHandler.SECTION_GENERAL,

    "atom": FFieldHandler.SECTION_ATOM,
    "atoms": FFieldHandler.SECTION_ATOM,

    "bond": FFieldHandler.SECTION_BOND,
    "bonds": FFieldHandler.SECTION_BOND,

    "off_diagonal": FFieldHandler.SECTION_OFF_DIAGONAL,
    "off-diagonal": FFieldHandler.SECTION_OFF_DIAGONAL,
    "offdiag": FFieldHandler.SECTION_OFF_DIAGONAL,

    "angle": FFieldHandler.SECTION_ANGLE,
    "angles": FFieldHandler.SECTION_ANGLE,

    "torsion": FFieldHandler.SECTION_TORSION,
    "torsions": FFieldHandler.SECTION_TORSION,

    "hbond": FFieldHandler.SECTION_HBOND,
    "hbonds": FFieldHandler.SECTION_HBOND,
    "hydrogen_bond": FFieldHandler.SECTION_HBOND,
    "hydrogen_bonds": FFieldHandler.SECTION_HBOND,
}


def _normalize_section_name(section: str) -> str:
    """Normalize user input like 'Off-Diagonal', ' off diag ' -> 'off_diagonal'."""
    return section.strip().lower().replace("-", "_").replace(" ", "_")


def get_ffield_data(handler: BaseHandler, *, section: str) -> pd.DataFrame:
    """
    Retrieve a specific section of the ReaxFF force-field file as a DataFrame.

    Works on
    --------
    FFieldHandler — ``ffield``

    Parameters
    ----------
    handler : FFieldHandler
        Parsed force-field handler.
    section : str
        Section name to retrieve (e.g. ``atom``, ``bond``, ``off_diagonal``,
        ``angle``, ``torsion``, ``hbond``). Aliases are supported.

    Returns
    -------
    pandas.DataFrame
        Table of parameters for the requested section.

    Examples
    --------
    >>> from reaxkit.io.handlers.ffield_handler import FFieldHandler
    >>> from reaxkit.analysis.per_file.ffield_analyzer import get_ffield_data
    >>> h = FFieldHandler("ffield")
    >>> df = get_ffield_data(h, section="bond")
    """

    # Type check
    if not isinstance(handler, FFieldHandler):
        raise TypeError(
            f"get_sections_data requires an FFieldHandler, got {type(handler)!r}"
        )

    # Normalize the name
    norm = _normalize_section_name(section)

    if norm not in _SECTION_ALIASES:
        raise KeyError(
            f"Unknown section {section!r}. Valid options: {sorted(_SECTION_ALIASES.keys())}"
        )

    # Resolve canonical key
    canonical = _SECTION_ALIASES[norm]

    # Use your kept section_df method
    try:
        df = handler.section_df(canonical)
    except KeyError:
        raise KeyError(
            f"Section {section!r} (canonical key {canonical!r}) not found in this ffield."
        )

    # Always return a copy
    return df.copy()

###############################################################################
# A “interpret indices → atom symbols” utility.
# It uses the atom section to build an index -> symbol map for bond,
# off-diagonal, angle, torsion, hbond.
# This is used to understand what 1 1 mean in bond data, which shows C-C
# bond if C is the atom number 1 in the ffield.
###############################################################################

def _atom_index_to_symbol_map(handler: FFieldHandler) -> Dict[int, str]:
    """
    Build {atom_index: symbol} from the atom section.

    Notes:
    - atom_df index is 'atom_index' (1-based in your parser).
    - symbol column is expected to be like 'C', 'H', 'O', ...
    """
    atom_df = handler.section_df(FFieldHandler.SECTION_ATOM)

    if "symbol" not in atom_df.columns:
        raise KeyError(
            "Atom section does not contain a 'symbol' column. "
            "Check FFieldHandler._parse_atom_section()."
        )

    out: Dict[int, str] = {}
    for idx, row in atom_df.iterrows():
        try:
            i = int(idx)
        except Exception:
            continue
        sym = row.get("symbol")
        if sym is None or (isinstance(sym, float) and pd.isna(sym)) or str(sym).strip() == "":
            sym = f"atom{i}"
        out[i] = str(sym).strip()
    return out


def _add_symbols_for_columns(
    df: pd.DataFrame,
    idx_to_sym: Dict[int, str],
    cols: Sequence[str],
    *,
    term_col: str = "term",
    sep: str = "-",
) -> pd.DataFrame:
    """
    For each integer index column in `cols` (e.g. ['i','j','k']),
    add '<col>_symbol' columns and a combined `term` column.
    """
    out = df.copy()

    sym_cols: list[str] = []
    for c in cols:
        sc = f"{c}_symbol"
        sym_cols.append(sc)

        def _map_one(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            try:
                iv = int(v)
            except Exception:
                return None
            return idx_to_sym.get(iv, f"atom{iv}")

        out[sc] = out[c].map(_map_one)

    # Build readable label: e.g. C-H, C-C-C, O-H-O
    out[term_col] = out[sym_cols].apply(
        lambda r: sep.join([x for x in r.tolist() if x is not None]),
        axis=1,
    )
    return out


def interpret_ffield_terms(
    handler: FFieldHandler,
    *,
    sections: Optional[Iterable[str]] = None,
    sep: str = "-",
) -> Dict[str, pd.DataFrame]:
    """
    Interpret numeric atom indices in ffield sections into chemical symbols.

    Works on
    --------
    FFieldHandler — ``ffield``

    Parameters
    ----------
    handler : FFieldHandler
        Parsed force-field handler.
    sections : iterable of str, optional
        Sections to interpret. Default interprets all relevant sections:
        ``bond``, ``off_diagonal``, ``angle``, ``torsion``, ``hbond``.
    sep : str, default="-"
        Separator used when building human-readable interaction labels.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping from section name to interpreted DataFrame. Each DataFrame
        includes:
        - ``*_symbol`` columns (e.g. ``i_symbol``, ``j_symbol``)
        - ``term`` column with readable labels (e.g. ``C-H``, ``C-C-C``)

    Examples
    --------
    >>> from reaxkit.io.handlers.ffield_handler import FFieldHandler
    >>> from reaxkit.analysis.per_file.ffield_analyzer import interpret_ffield_terms
    >>> h = FFieldHandler("ffield")
    >>> data = interpret_ffield_terms(h, sections=["bond", "angle"])
    >>> bond_df = data["bond"]
    """
    if not isinstance(handler, FFieldHandler):
        raise TypeError(f"interpret_ffield_terms requires FFieldHandler, got {type(handler)!r}")

    idx_to_sym = _atom_index_to_symbol_map(handler)

    targets = [
        FFieldHandler.SECTION_BOND,
        FFieldHandler.SECTION_OFF_DIAGONAL,
        FFieldHandler.SECTION_ANGLE,
        FFieldHandler.SECTION_TORSION,
        FFieldHandler.SECTION_HBOND,
    ]
    if sections is not None:
        wanted = {s.strip().lower() for s in sections}
        targets = [t for t in targets if t in wanted]

    out: Dict[str, pd.DataFrame] = {}

    for sec in targets:
        df = handler.section_df(sec).copy()

        if sec in (FFieldHandler.SECTION_BOND, FFieldHandler.SECTION_OFF_DIAGONAL):
            # columns: i, j
            out[sec] = _add_symbols_for_columns(df, idx_to_sym, ["i", "j"], sep=sep)
        elif sec == FFieldHandler.SECTION_ANGLE:
            # columns: i, j, k
            out[sec] = _add_symbols_for_columns(df, idx_to_sym, ["i", "j", "k"], sep=sep)
        elif sec == FFieldHandler.SECTION_TORSION:
            # columns: i, j, k, l
            out[sec] = _add_symbols_for_columns(df, idx_to_sym, ["i", "j", "k", "l"], sep=sep)
        elif sec == FFieldHandler.SECTION_HBOND:
            # columns: i, j, k  (commonly X-H-Y order in your example)
            out[sec] = _add_symbols_for_columns(df, idx_to_sym, ["i", "j", "k"], sep=sep)
        else:
            out[sec] = df

    return out


def interpret_one_section(
    handler: FFieldHandler,
    *,
    section: str,
    sep: str = "-",
) -> pd.DataFrame:
    """
    Interpret a single ffield section into symbol-based interaction labels.

    Works on
    --------
    FFieldHandler — ``ffield``

    Parameters
    ----------
    handler : FFieldHandler
        Parsed force-field handler.
    section : str
        Section to interpret (e.g. ``bond``, ``angle``, ``torsion``, ``hbond``).
    sep : str, default="-"
        Separator used when building the ``term`` label.

    Returns
    -------
    pandas.DataFrame
        Interpreted section DataFrame with ``*_symbol`` columns and a
        human-readable ``term`` column.

    Examples
    --------
    >>> from reaxkit.io.handlers.ffield_handler import FFieldHandler
    >>> from reaxkit.analysis.per_file.ffield_analyzer import interpret_one_section
    >>> h = FFieldHandler("ffield")
    >>> bond_df = interpret_one_section(h, section="bond")
    """
    section = section.strip().lower().replace("-", "_").replace(" ", "_")
    res = interpret_ffield_terms(handler, sections=[section], sep=sep)
    if section not in res:
        raise KeyError(f"Section {section!r} not found or not supported for interpretation.")
    return res[section]
