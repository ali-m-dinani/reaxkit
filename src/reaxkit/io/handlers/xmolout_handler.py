"""
ReaxFF trajectory output (xmolout) handler.

This module provides a handler for parsing ReaxFF ``xmolout`` files,
which store atomic trajectories from MD runs or MM minimizations.

``xmolout`` files contain repeated coordinate frames with associated
cell parameters and energies and are commonly used for visualization
and structural analysis.
"""


from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any
import pandas as pd
from reaxkit.io.base_handler import FileHandler

class XmoloutHandler(FileHandler):
    """
    Parser for ReaxFF trajectory output files (``xmolout``).

    This class parses ``xmolout`` files and exposes both a per-frame
    summary table and per-frame atomic coordinate tables.

    Parsed Data
    -----------
    Summary table
        One row per frame, returned by ``dataframe()``, with columns:
        ["num_of_atoms", "iter", "E_pot",
         "a", "b", "c", "alpha", "beta", "gamma"]

        Duplicate iteration indices are removed by keeping the last
        occurrence.

    Per-frame atom tables
        Stored in ``self._frames``, one table per frame, where each table
        has at least the columns:
        ["atom_type", "x", "y", "z"]

        Any additional per-atom columns present in the file are preserved
        per frame. If their names are not provided explicitly, they are
        auto-named as ``unknown_1``, ``unknown_2``, …

    Metadata
        Returned by ``metadata()``, containing:
        ["simulation_name", "n_atoms", "n_frames", "has_time"]

    Notes
    -----
    - Frames are inferred from the repeating ``#atoms → header → atoms`` pattern.
    - The number of atoms is assumed constant across all frames.
    - This handler supports lightweight frame access via ``frame(i)``
      and streaming access via ``iter_frames(step=...)``.
    """

    def __init__(self, file_path: str | Path = "xmolout", *, extra_atom_cols: Optional[list[str]] = None):
        super().__init__(file_path)
        self._frames: List[pd.DataFrame] = []     # list of per-frame atom tables
        self._n_atoms: Optional[int] = None
        self.simulation_name: str = ""
        self._extra_atom_cols = list(extra_atom_cols) if extra_atom_cols else None

    # ---- FileHandler requirement
    def _parse(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        sim_rows: List[list] = []
        frames: List[pd.DataFrame] = []

        sim_cols = ["num_of_atoms", "iter", "E_pot", "a", "b", "c", "alpha", "beta", "gamma"]
        base_atom_cols = ["atom_type", "x", "y", "z"]

        with open(self.path, "r") as fh:
            atom_buf: List[list] = []
            atom_count = 0
            current_atom_cols: Optional[List[str]] = None
            n_atoms: Optional[int] = None

            for line in fh:
                vals = line.strip().split()
                if not vals:
                    continue

                # #atoms line
                if len(vals) == 1 and vals[0].isdigit():
                    n_atoms = int(vals[0])
                    self._n_atoms = n_atoms
                    sim_rows.append([n_atoms])  # placeholder row; will complete after header line
                    atom_buf, atom_count = [], 0
                    current_atom_cols = None
                    continue

                # header line (name iter E a b c alpha beta gamma)
                if len(vals) == 9 and self._n_atoms and vals[1].lstrip("-").isdigit():
                    if not self.simulation_name:
                        self.simulation_name = vals[0]
                    row = [self._n_atoms, int(vals[1])] + list(map(float, vals[2:]))
                    sim_rows[-1] = row
                    continue

                # atom coordinates (optionally with extra columns)
                if self._n_atoms and len(vals) >= 4:
                    # lazily determine expected columns for this frame
                    if current_atom_cols is None:
                        n_extras = max(0, len(vals) - 4)
                        if self._extra_atom_cols:
                            names = list(self._extra_atom_cols)[:n_extras]
                            if len(names) < n_extras:
                                names += [f"unknown_{i+1}" for i in range(n_extras - len(names))]
                        else:
                            names = [f"unknown_{i+1}" for i in range(n_extras)]
                        current_atom_cols = base_atom_cols + names

                    base = [vals[0]] + list(map(float, vals[1:4]))
                    expected_extras = len(current_atom_cols) - 4
                    extras_vals = [float(x) for x in vals[4:4+expected_extras]]
                    # pad if fewer provided
                    while len(extras_vals) < expected_extras:
                        extras_vals.append(float('nan'))
                    atom_buf.append(base + extras_vals)
                    atom_count += 1

                    if atom_count == self._n_atoms:
                        frames.append(pd.DataFrame(atom_buf, columns=current_atom_cols))
                        atom_buf, atom_count = [], 0
                        current_atom_cols = None

        # Build per-frame summary table
        df = pd.DataFrame(sim_rows, columns=sim_cols)

        # Deduplicate by iter (keep last)
        if not df.empty and "iter" in df.columns:
            keep_idx = df.drop_duplicates("iter", keep="last").index
            frames = [frames[i] for i in keep_idx if i < len(frames)]
            df = df.loc[keep_idx].reset_index(drop=True)

        # Save frames
        self._frames = frames

        meta: Dict[str, Any] = {
            "simulation_name": self.simulation_name,
            "n_atoms": self._n_atoms,
            "n_frames": len(self._frames),
            "has_time": False,
        }
        return df, meta

    # ---- Explicit, file-specific accessors (no generic get())
    def n_frames(self) -> int:
        return int(self.metadata().get("n_frames", 0))

    def n_atoms(self) -> Optional[int]:
        return self._n_atoms

    def frame(self, i: int) -> Dict[str, Any]:
        """Return a lightweight frame dict: coords + atom_types + iter for frame i."""
        df = self.dataframe()
        if i < 0 or i >= len(self._frames):
            raise IndexError(f"frame index {i} out of range [0, {len(self._frames) - 1}]")

        frame_df = self._frames[i]
        coords = frame_df[["x", "y", "z"]].to_numpy(dtype=float)
        atom_types = frame_df["atom_type"].astype(str).tolist()

        row = df.iloc[i]
        return {
            "index": i,
            "iter": int(row["iter"]) if "iter" in df.columns else i,
            "coords": coords,
            "atom_types": atom_types,
        }

    def iter_frames(self, step: int = 1) -> Iterator[Dict[str, Any]]:
        for i in range(0, self.n_frames(), max(1, int(step))):
            yield self.frame(i)
