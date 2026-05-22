"""
Geometry structure I/O utilities.

This module contains ASE-based helpers for reading and writing structure files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ase import Atoms
from ase.io import read, write


__all__ = ["read_structure", "write_structure"]


def read_structure(
    path: str | Path,
    format: Optional[str] = None,
    index: int | str = 0,
) -> Atoms:
    """
    Read a structure file using ASE.
    """
    path = Path(path)
    return read(path, format=format, index=index)


def write_structure(
    atoms: Atoms,
    path: str | Path,
    format: Optional[str] = None,
    comment: Optional[str] = None,
) -> Path:
    """
    Write a structure to file in any ASE-supported format.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"format": format}
    if comment is not None:
        kwargs["comment"] = comment
    write(path, atoms, **kwargs)
    return path
