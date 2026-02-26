"""
Physical and numerical constants utilities for ReaxKit.

This module provides access to commonly used constants defined in a packaged
YAML file and exposes a small, cached API for retrieving them by name.

Constant values are stored in ``reaxkit/data/constants.yaml`` and loaded on
demand.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

import yaml
import importlib.resources as ir


@lru_cache(maxsize=1)
def _load_constants() -> Dict[str, float]:
    """
    Load packaged constants into a dictionary.

    Constants are read from ``constants.yaml`` and cached after the first call
    to avoid repeated disk access.

    Returns
    -------
    dict[str, float]
        Mapping of constant names to numeric values.

    Raises
    ------
    FileNotFoundError
        If the packaged constants file cannot be located.
    """
    pkg = "reaxkit"
    rel = "data/constants.yaml"

    try:
        with ir.files(pkg).joinpath(rel).open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find packaged constants at '{pkg}/{rel}'. "
            "Make sure constants.yaml is included as package data."
        ) from e

    raw = doc.get("constants") or {}
    return {str(k): float(v) for k, v in raw.items()}


def const(name: str, default: Optional[float] = None) -> Optional[float]:
    """
    Retrieve a named constant.

    Parameters
    ----------
    name : str
        Name of the constant to retrieve.
    default : float, optional
        Value to return if the constant is not defined.

    Returns
    -------
    float or None
        Constant value if found; otherwise ``default``.

    Examples
    --------
    >>> const("kB")
    >>> const("e_charge", default=1.0)
    """
    return _load_constants().get(name, default)
