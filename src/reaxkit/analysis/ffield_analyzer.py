# reaxkit/analysis/ffield_analyzer.py
from __future__ import annotations

from typing import Dict

import pandas as pd

from reaxkit.io.template_handler import TemplateHandler
from reaxkit.io.ffield_handler import FFieldHandler


# Map user-friendly section names to FFieldHandler section keys
_SECTION_ALIASES: Dict[str, str] = {
    # general
    "general": FFieldHandler.SECTION_GENERAL,

    # atoms
    "atom": FFieldHandler.SECTION_ATOM,
    "atoms": FFieldHandler.SECTION_ATOM,
    "atom_parameters": FFieldHandler.SECTION_ATOM,
    "atom_params": FFieldHandler.SECTION_ATOM,

    # bonds
    "bond": FFieldHandler.SECTION_BOND,
    "bonds": FFieldHandler.SECTION_BOND,
    "bond_parameters": FFieldHandler.SECTION_BOND,
    "bond_params": FFieldHandler.SECTION_BOND,

    # off-diagonal
    "off_diagonal": FFieldHandler.SECTION_OFF_DIAGONAL,
    "off-diagonal": FFieldHandler.SECTION_OFF_DIAGONAL,
    "offdiag": FFieldHandler.SECTION_OFF_DIAGONAL,
    "off_diagonal_parameters": FFieldHandler.SECTION_OFF_DIAGONAL,
    "off_diagonal_params": FFieldHandler.SECTION_OFF_DIAGONAL,

    # angles
    "angle": FFieldHandler.SECTION_ANGLE,
    "angles": FFieldHandler.SECTION_ANGLE,
    "angle_parameters": FFieldHandler.SECTION_ANGLE,
    "angle_params": FFieldHandler.SECTION_ANGLE,

    # torsions
    "torsion": FFieldHandler.SECTION_TORSION,
    "torsions": FFieldHandler.SECTION_TORSION,
    "torsion_parameters": FFieldHandler.SECTION_TORSION,
    "torsion_params": FFieldHandler.SECTION_TORSION,

    # hydrogen bonds
    "hbond": FFieldHandler.SECTION_HBOND,
    "hbonds": FFieldHandler.SECTION_HBOND,
    "hydrogen_bond": FFieldHandler.SECTION_HBOND,
    "hydrogen_bonds": FFieldHandler.SECTION_HBOND,
    "hbond_parameters": FFieldHandler.SECTION_HBOND,
    "hbond_params": FFieldHandler.SECTION_HBOND,
}


def _normalize_section_name(section: str) -> str:
    """Normalize a user-provided section name to a canonical key."""
    key = section.strip().lower().replace("-", "_").replace(" ", "_")
    return key


def get_sections_data(handler: TemplateHandler, *, section: str) -> pd.DataFrame:
    """
    Return a section DataFrame from an FFieldHandler.

    Parameters
    ----------
    handler : TemplateHandler
        Should be an instance of FFieldHandler (or subclass).
    section : str
        Name of the section to retrieve. Accepted values (case-insensitive):

        - "general"
        - "atom", "atoms"
        - "bond", "bonds"
        - "off_diagonal", "off-diagonal", "offdiag"
        - "angle", "angles"
        - "torsion", "torsions"
        - "hbond", "hbonds", "hydrogen_bond", "hydrogen_bonds"

    Returns
    -------
    pandas.DataFrame
        The requested section DataFrame (a copy).

    Raises
    ------
    TypeError
        If `handler` is not an FFieldHandler.
    KeyError
        If the requested section is not recognized or not present.
    """
    if not isinstance(handler, FFieldHandler):
        raise TypeError(
            f"ffield_analyzer.get expects an FFieldHandler, got {type(handler)!r}"
        )

    norm = _normalize_section_name(section)

    if norm not in _SECTION_ALIASES:
        raise KeyError(
            f"Unknown ffield section {section!r}. "
            f"Valid options: {sorted(_SECTION_ALIASES.keys())}"
        )

    section_key = _SECTION_ALIASES[norm]

    # This will trigger parsing if needed
    df = handler.section_df(section_key)

    # Return a copy so downstream code doesn't accidentally mutate handler's cache
    return df.copy()
