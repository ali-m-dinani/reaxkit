"""analyzer for ffield file"""

from __future__ import annotations
from typing import Dict

import pandas as pd

from reaxkit.io.template_handler import TemplateHandler
from reaxkit.io.ffield_handler import FFieldHandler


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


def get_sections_data(handler: TemplateHandler, *, section: str) -> pd.DataFrame:
    """
    Retrieve a specific ffield section using handler.section_df().
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
