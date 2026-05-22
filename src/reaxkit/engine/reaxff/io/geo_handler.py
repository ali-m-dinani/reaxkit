"""
ReaxFF geometry structure (geo) file handler.

This module provides a handler for parsing ReaxFF ``.geo`` structure
files in XTLGRF format, which define atomic coordinates, optional
periodic cell parameters, and descriptive metadata for a system.

Typical use cases include:

- loading initial or relaxed geometries
- extracting atomic coordinates for analysis or visualization
- accessing unit cell parameters for periodic systems
"""


from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd

from reaxkit.engine.reaxff.io.base import BaseHandler


class GeoHandler(BaseHandler):
    """
    Parser for ReaxFF geometry structure files (``.geo`` / XTLGRF format).

    This class parses ``.geo`` files and exposes atomic coordinates and
    associated structural metadata as structured Python objects.

    Parsed Data
    -----------
    Atom table
        One row per atom, returned by ``dataframe()``, with columns:
        ["atom_id", "atom_type", "x", "y", "z"]

    Connectivity table
        One row per declared CONECT edge, returned by ``connectivity()``,
        with columns:
        ["source_atom_id", "target_atom_id"]

    Metadata
        Returned by ``metadata()``, containing:
        {
            "descriptor": str | None,     # from DESCRP line
            "remark": str | None,         # concatenated REMARK lines
            "cell_lengths": {             # from CRYSTX (a, b, c)
                "a": float,
                "b": float,
                "c": float,
            } | None,
            "cell_angles": {              # from CRYSTX (alpha, beta, gamma)
                "alpha": float,
                "beta": float,
                "gamma": float,
            } | None,
            "n_atoms": int,
        }

    Notes
    -----
    - Only ``ATOM`` and ``HETATM`` records are parsed into the atom table.
    - ``CONECT`` records (after ``FORMAT CONECT``) are parsed into a
      separate connectivity table.
    - Cell parameters are optional and may be absent for non-periodic systems.
    - Non-structural lines (e.g. ``XTLGRF``, ``FORMAT``) are ignored.
    - This handler is not frame-based; the file represents a single structure.
    """

    def __init__(self, file_path: str | Path = "geo", reporter=None):
        """
        Initialize the instance.

        Parameters
        ----------
        file_path : str | Path
            Parameter description.

        """
        super().__init__(file_path)
        self._n_atoms: Optional[int] = None
        self._connectivity_df: Optional[pd.DataFrame] = None
        self._reporter = reporter

    # ------------------------------------------------------------------
    # Core parser
    # ------------------------------------------------------------------
    def _parse(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
         parse.

        Returns
        -------
        tuple[pd.DataFrame, dict[str, Any]]
            Return value description.

        """
        atoms: List[Dict[str, Any]] = []
        connectivity_rows: List[Dict[str, int]] = []

        descriptor: Optional[str] = None
        remark: Optional[str] = None
        cell_lengths: Optional[Dict[str, float]] = None
        cell_angles: Optional[Dict[str, float]] = None
        in_conect_section = False

        total_lines = self._count_lines()
        with open(self.path, "r") as fh:
            lines_read = 0
            for raw in fh:
                lines_read += 1
                if self._reporter and (lines_read % 1000 == 0 or lines_read == total_lines):
                    self._reporter("load", lines_read, total_lines, "Parsing geo")
                line = raw.rstrip("\n")
                stripped = line.strip()
                if not stripped:
                    continue

                # Enter/leave CONECT block.
                if stripped.upper().startswith("FORMAT CONECT"):
                    in_conect_section = True
                    continue
                if stripped.upper().startswith("END"):
                    in_conect_section = False
                    continue

                # Connectivity records: CONECT src dst1 dst2 ...
                if in_conect_section and stripped.upper().startswith("CONECT"):
                    parts = stripped.split()
                    if len(parts) >= 2:
                        try:
                            source_atom_id = int(parts[1])
                        except ValueError:
                            continue
                        for token in parts[2:]:
                            try:
                                target_atom_id = int(token)
                            except ValueError:
                                continue
                            connectivity_rows.append(
                                {
                                    "source_atom_id": source_atom_id,
                                    "target_atom_id": target_atom_id,
                                }
                            )
                    continue

                # Descriptor
                if line.startswith("DESCRP"):
                    # everything after the keyword is the descriptor
                    # "DESCRP" is 6 chars, keep the rest
                    text = line[6:].strip()
                    if not text:
                        # fallback: split-based if for some reason slicing fails
                        parts = line.split(maxsplit=1)
                        text = parts[1].strip() if len(parts) > 1 else ""
                    descriptor = text or None
                    continue

                # Remark (optional, possibly multiple lines)
                if line.startswith("REMARK"):
                    text = line[6:].strip()
                    if remark:
                        remark = f"{remark} {text}".strip()
                    else:
                        remark = text
                    continue

                # Cell / periodic box
                if line.startswith("CRYSTX"):
                    # Expected: CRYSTX a b c alpha beta gamma
                    parts = line.split()
                    if len(parts) >= 7:
                        try:
                            a, b, c = map(float, parts[1:4])
                            alpha, beta, gamma = map(float, parts[4:7])
                            cell_lengths = {"a": a, "b": b, "c": c}
                            cell_angles = {
                                "alpha": alpha,
                                "beta": beta,
                                "gamma": gamma,
                            }
                        except ValueError:
                            # If parsing fails, leave as None
                            pass
                    continue

                # Atom records: HETATM or ATOM
                if line.startswith("HETATM") or line.startswith("ATOM"):
                    parts = line.split()
                    # We expect at least:
                    #   0: "HETATM" / "ATOM"
                    #   1: atom_id (int)
                    #   2: atom_type (str, e.g., N, Al, O_w, ...)
                    #   3: x
                    #   4: y
                    #   5: z
                    #   6: repeated atom type (ignored)
                    #   7+: extra fields (ignored)
                    if len(parts) < 7:
                        # Too short to contain id, type, and coordinates
                        continue

                    try:
                        atom_id = int(parts[1])
                    except ValueError:
                        # Unexpected format, skip this line
                        continue

                    atom_type = parts[2]
                    try:
                        x, y, z = map(float, parts[3:6])
                    except ValueError:
                        # Coordinates not parseable, skip
                        continue

                    atoms.append(
                        {
                            "atom_id": atom_id,
                            "atom_type": atom_type,
                            "x": x,
                            "y": y,
                            "z": z,
                        }
                    )

                # Other lines (XTLGRF, FORMAT, etc.) are ignored

        df = pd.DataFrame(atoms, columns=["atom_id", "atom_type", "x", "y", "z"])
        self._connectivity_df = pd.DataFrame(
            connectivity_rows,
            columns=["source_atom_id", "target_atom_id"],
        )
        n_atoms = len(df)
        self._n_atoms = n_atoms

        meta: Dict[str, Any] = {
            "descriptor": descriptor,
            "remark": remark,
            "cell_lengths": cell_lengths,
            "cell_angles": cell_angles,
            "n_atoms": n_atoms,
            "n_connectivity_edges": int(len(self._connectivity_df)),
        }
        if self._reporter:
            self._reporter("load", total_lines, total_lines, "Finished parsing geo")
        return df, meta

    def _count_lines(self) -> int:
        with open(self.path, "r") as fh:
            return sum(1 for _ in fh)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def n_atoms(self) -> int:
        """
        N atoms.

        Returns
        -------
        int
            Return value description.

        """
        if self._n_atoms is None:
            self._n_atoms = int(self.metadata().get("n_atoms", len(self.dataframe())))
        return self._n_atoms

    def cell(self) -> Dict[str, Optional[float]]:
        """
        Return a flat dict with cell parameters:

            {
                "a": ...,
                "b": ...,
                "c": ...,
                "alpha": ...,
                "beta": ...,
                "gamma": ...,
            }

        Values may be None if CRYSTX was missing or malformed.
        """
        meta = self.metadata()
        lengths = meta.get("cell_lengths") or {}
        angles = meta.get("cell_angles") or {}

        return {
            "a": lengths.get("a"),
            "b": lengths.get("b"),
            "c": lengths.get("c"),
            "alpha": angles.get("alpha"),
            "beta": angles.get("beta"),
            "gamma": angles.get("gamma"),
        }

    def coordinates(self) -> pd.DataFrame:
        """
        Return a copy of the atom table (id, type, x, y, z).

        This is just a convenience wrapper around .dataframe()
        to make the intent explicit.
        """
        return self.dataframe().copy()

    def connectivity(self) -> pd.DataFrame:
        """
        Return a copy of the parsed CONECT edge table.

        Returns
        -------
        pd.DataFrame
            Columns:
            - source_atom_id
            - target_atom_id
        """
        if not self._parsed:
            self.parse()
        if self._connectivity_df is None:
            return pd.DataFrame(columns=["source_atom_id", "target_atom_id"])
        return self._connectivity_df.copy()
