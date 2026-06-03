"""Read and write geometry structures through ASE-backed helpers.

This module provides thin, engine-agnostic wrappers around ``ase.io.read`` and
``ase.io.write``. It focuses on simple path handling and output-directory
creation while leaving format parsing/serialization to ASE.

**Usage context**

- Common I/O layer: Shared by generators and structure-transformer utilities.
- Format interoperability: Supports any structure format recognized by ASE.
- File safety: Ensures parent directories exist before writing outputs.
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
    """Read a structure file using ASE.

    Parameters
    ----------
    path : str | Path
        Input structure file path.
    format : Optional[str], optional
        Explicit ASE format override. ``None`` lets ASE infer the format.
    index : int | str, optional
        Structure index selection forwarded to ``ase.io.read``.

    Returns
    -------
    Atoms
        Parsed ASE ``Atoms`` object.

    Examples
    --------
    ```python
    atoms = read_structure("slab.xyz")
    ```
    """
    path = Path(path)
    return read(path, format=format, index=index)


def write_structure(
    atoms: Atoms,
    path: str | Path,
    format: Optional[str] = None,
    comment: Optional[str] = None,
) -> Path:
    """Write an ASE structure file in any ASE-supported format.

    Parameters
    ----------
    atoms : Atoms
        Structure object to serialize.
    path : str | Path
        Output file path.
    format : Optional[str], optional
        Explicit ASE format override. ``None`` lets ASE infer from filename.
    comment : Optional[str], optional
        Optional comment string forwarded to ASE writers that support it.

    Returns
    -------
    Path
        Normalized output path written to disk.

    Examples
    --------
    ```python
    out = write_structure(atoms, "outputs/model.xyz", comment="generated")
    ```
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"format": format}
    if comment is not None:
        kwargs["comment"] = comment
    write(path, atoms, **kwargs)
    return path
