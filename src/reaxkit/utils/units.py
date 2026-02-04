"""
Unit metadata utilities for ReaxKit quantities.

This module provides access to display units associated with canonical
quantity keys used across ReaxFF files and ReaxKit analyses.

Unit definitions are stored in the packaged file
``reaxkit/data/units.yaml`` and loaded on demand.
"""


from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

import yaml
import importlib.resources as ir


@lru_cache(maxsize=1)
def _load_units_map() -> Dict[str, str]:
    """
    Load the canonical key → unit mapping.

    Units are read from the packaged ``units.yaml`` file and cached after
    the first call to avoid repeated disk access.

    Returns
    -------
    dict[str, str]
        Mapping of canonical quantity keys to display units.

    Raises
    ------
    FileNotFoundError
        If the packaged units file cannot be located.
    """
    pkg = "reaxkit"
    rel = "data/units.yaml"

    try:
        with ir.files(pkg).joinpath(rel).open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find packaged units map at '{pkg}/{rel}'. "
            "Make sure units.yaml is included as package data."
        ) from e

    raw = doc.get("units") or {}
    return {str(k): str(v) for k, v in raw.items() if v is not None}


def unit_for(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieve the display unit for a canonical quantity key.

    Parameters
    ----------
    key : str
        Canonical quantity key (e.g., ``"energy"``, ``"time"``).
    default : str, optional
        Unit string to return if the key is not defined.

    Returns
    -------
    str or None
        Unit string if found; otherwise ``default``.

    Examples
    --------
    >>> unit_for("energy")
    >>> unit_for("pressure", default="MPa")
    """
    return _load_units_map().get(str(key), default)


def _label_with_unit(label: str, key: str) -> str:
    """
    Construct a label string with an associated unit.

    If a unit exists for the given key, it is appended in parentheses;
    otherwise, the label is returned unchanged.

    Parameters
    ----------
    label : str
        Base label text (e.g., ``"Energy"``).
    key : str
        Canonical quantity key used to look up the unit.

    Returns
    -------
    str
        Label with unit appended if available.

    Examples
    --------
    >>> _label_with_unit("Energy", "energy")
    >>> _label_with_unit("Time", "time")
    """
    u = unit_for(key)
    return f"{label} ({u})" if u else label
