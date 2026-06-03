"""
ReaxFF connectivity (fort.7) file handler.

This module provides a handler for parsing ReaxFF ``fort.7`` files,
which store per-iteration atom connectivity, bond-order information,
and system-wide totals.

Typical use cases include:

- extracting per-atom bond-order features
- computing coordination statistics
- building molecule- and structure-level descriptors

**Usage context**

- ReaxFF parsing: Read ReaxFF text outputs into normalized tabular structures.
- Workflow ingestion: Provide canonical handler interfaces used by adapters/workflows.
- Diagnostics/export: Preserve parsed metadata for reporting and downstream conversion.
"""


from __future__ import annotations
import json
from pathlib import Path
import pickle
import shutil
from typing import List, Dict, Any, Optional
import pandas as pd

from reaxkit.engine.reaxff.io.base import BaseHandler


class Fort7Handler(BaseHandler):
    """
    Parser for ReaxFF connectivity output files (``fort.7``).

    This class parses ReaxFF ``fort.7`` files and exposes both
    iteration-level summaries and per-iteration atom connectivity
    tables as structured tabular data.

    Parsed Data
    -----------
    Summary table
        One row per iteration, returned by ``dataframe()``, with columns:
        ["iter", "num_of_atoms", "num_of_bonds",
         "total_BO", "total_LP", "total_BO_uncorrected", "total_charge"]

    Per-frame atom tables
        Stored in ``self._frames``, one table per iteration, where each
        frame is a ``pandas.DataFrame`` with columns:
        ["atom_num", "atom_type_num", "atom_cnn1..nb", "molecule_num",
         "BO1..nb", "sum_BOs", "num_LPs", "partial_charge", ...]

        Here, ``nb`` denotes the number of bonded neighbors in that frame,
        leading to variable-length connectivity and bond-order columns.

    Metadata
        Returned by ``metadata()``, containing:
        ["n_frames", "n_records", "simulation_name"]

    Notes
    -----
    - Duplicate iterations are resolved by keeping the last occurrence.
    - Connectivity and bond-order columns are inferred from the header.
    - Extra, file-dependent columns are preserved as ``unknown*`` fields.
    """
    def __init__(self, file_path: str | Path = "fort.7", reporter=None):
        """Initialize a handler for a ReaxFF ``fort.7`` connectivity file.

        Works on
        --------
        Fort7Handler — ``fort.7``

        Parameters
        ----------
        file_path : str or pathlib.Path, optional
            Path to the ``fort.7`` file to be parsed.

        Returns
        -------
        None
            Initializes the handler without parsing the file.
        """
        super().__init__(file_path)
        self._frames: List[pd.DataFrame] = []
        self._sim_name: Optional[str] = None
        self._reporter = reporter

    def _parse(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
         parse.

        Returns
        -------
        tuple[pd.DataFrame, dict[str, Any]]
            Return value description.

        """
        sim_rows: List[List[Any]] = []
        frames: List[pd.DataFrame] = []
        totals: List[List[float]] = []

        cur_atoms_rows: List[List[float | int]] = []
        cur_totals: List[float] = []
        cur_num_particles: Optional[int] = None
        cur_nbonds: Optional[int] = None
        sim_name: str = ""
        warned_large_atom_count = False

        def _finalize_iteration() -> None:
            """Finalize iteration."""
            if cur_num_particles is None or cur_nbonds is None or not cur_atoms_rows:
                return
            nb = int(cur_nbonds)
            atom_cols = (
                ["atom_num", "atom_type_num"]
                + [f"atom_cnn{i}" for i in range(1, nb + 1)]
                + ["molecule_num"]
                + [f"BO{i}" for i in range(1, nb + 1)]
                + ["sum_BOs", "num_LPs", "partial_charge"]
            )
            extra = max(0, len(cur_atoms_rows[0]) - len(atom_cols))
            if extra > 0:
                atom_cols += [f"unknown{i}" for i in range(1, extra + 1)]
            frames.append(pd.DataFrame(cur_atoms_rows, columns=atom_cols))
            totals.append(cur_totals[:] if cur_totals else [float("nan")] * 4)

        total_lines = self._count_lines()
        with open(self.path, "r") as fh:
            lines_read = 0
            for raw in fh:
                lines_read += 1
                if self._reporter and (lines_read % 5000 == 0 or lines_read == total_lines):
                    self._reporter("load", lines_read, total_lines, "Parsing fort.7")
                values = raw.split()
                # print(f"line {lines_read} has {len(values)} values")
                if not values:
                    continue

                # Header
                if len(values) == 6:
                    if cur_atoms_rows:
                        _finalize_iteration()
                        cur_atoms_rows.clear()
                        cur_totals.clear()

                    cur_num_particles = int(values[0])
                    if cur_num_particles > 9999 and not warned_large_atom_count:
                        warning_msg = (
                            "Warning: fort.7 reports > 9999 atoms. ReaxFF fixed-width atom-index fields "
                            "can overflow at this size, which may concatenate neighbor indices and corrupt "
                            "fort.7 connectivity parsing. Consider running 'repair_fort7' before analysis."
                        )
                        print(warning_msg)
                        if self._reporter:
                            try:
                                self._reporter("warn", lines_read, total_lines, warning_msg)
                            except TypeError:
                                # Backward-compatible fallback for reporters that only handle load events.
                                self._reporter("load", lines_read, total_lines, warning_msg)
                        warned_large_atom_count = True
                    sim_name = values[1]
                    iteration = int(values[3])
                    cur_nbonds = int(values[5])
                    sim_rows.append([iteration, cur_num_particles, cur_nbonds])

                # Totals
                elif len(values) < 6:
                    cur_totals.extend(map(float, values))

                # Atom line
                else:
                    nb = int(cur_nbonds)
                    int_part = list(map(int, values[0: nb + 3]))
                    float_part = list(map(float, values[nb + 3:]))
                    cur_atoms_rows.append(int_part + float_part)

        # Final iter
        if cur_atoms_rows:
            _finalize_iteration()

        # Summary dataframe
        sim_df = pd.DataFrame(sim_rows, columns=["iter", "num_of_atoms", "num_of_bonds"])
        totals_df = pd.DataFrame(
            totals,
            columns=["total_BO", "total_LP", "total_BO_uncorrected", "total_charge"]
            if totals and len(totals[0]) == 4
            else [f"total_val{i}" for i in range(1, (len(totals[0]) if totals else 0) + 1)]
        )
        if not totals_df.empty:
            totals_df = totals_df.iloc[: len(sim_df)].reset_index(drop=True)
            sim_df = pd.concat([sim_df.reset_index(drop=True), totals_df], axis=1)

        # Deduplicate
        if not sim_df.empty and "iter" in sim_df.columns:
            keep_idx = sim_df.drop_duplicates("iter", keep="last").index
            frames = [frames[i] for i in keep_idx]
            sim_df = sim_df.loc[keep_idx].reset_index(drop=True)

        self._frames = frames
        self._sim_name = sim_name

        meta: Dict[str, Any] = {
            "n_frames": len(frames),
            "n_records": len(sim_df),
            "simulation_name": sim_name,
        }
        if self._reporter:
            self._reporter("load", total_lines, total_lines, "Finished parsing fort.7")

        return sim_df, meta

    def _count_lines(self) -> int:
        """Count lines."""
        with open(self.path, "r") as fh:
            return sum(1 for _ in fh)

    # ---- disk-cache override (parquet + json) -------------------
    def _disk_cache_dir(self, key: str) -> Path:
        """Disk cache dir."""
        return self._cache_root() / key

    def _store_in_disk_cache(self, key: str, payload: bytes) -> None:
        """Store in disk cache."""
        super()._store_in_disk_cache(key, payload)

    def _load_from_disk_cache(self, key: str) -> bytes | None:
        """Load from disk cache."""
        return super()._load_from_disk_cache(key)

    # -------------------------------------------------------
    # Frame utilities (match XmoloutHandler API)
    # -------------------------------------------------------

    def n_frames(self) -> int:
        """
        Return the number of frames parsed from the ``fort.7`` file.

        Works on
        --------
        Fort7Handler — ``fort.7``

        Returns
        -------
        int
            Number of parsed frames (iterations).
        """
        return len(self._frames) if hasattr(self, "_frames") else 0

    def n_atoms(self, frame: int = 0) -> int:
        """
        Return the number of atoms in a given frame.

        Works on
        --------
        Fort7Handler — ``fort.7``

        Parameters
        ----------
        frame : int, optional
            Frame index to query.

        Returns
        -------
        int
            Number of atoms in the selected frame.
        """
        if not hasattr(self, "_frames") or self.n_frames() == 0:
            return 0
        return len(self._frames[int(frame)])

    def frame(self, i: int):
        """Return a single frame as an atom-level connectivity table.

        Works on
        --------
        Fort7Handler — ``fort.7``

        Parameters
        ----------
        i : int
            Frame index to retrieve.

        Returns
        -------
        pandas.DataFrame
            Atom-level table for the selected frame, including connectivity
            and bond-order columns.

        Examples
        --------
        >>> h = Fort7Handler("fort.7")
        >>> df = h.frame(0)
        """
        if not hasattr(self, "_frames"):
            raise RuntimeError("fort.7 has not been parsed yet.")
        return self._frames[int(i)]

    def iter_frames(self, step: int = 1):
        """Iterate over atom-level frames with optional subsampling.

        Works on
        --------
        Fort7Handler — ``fort.7``

        Parameters
        ----------
        step : int, optional
            Step size for subsampling frames (default: 1).

        Yields
        ------
        pandas.DataFrame
            Atom-level connectivity table for each yielded frame.

        Examples
        --------
        >>> h = Fort7Handler("fort.7")
        >>> for frame in h.iter_frames(step=10):
        ...     print(len(frame))
        """
        if not hasattr(self, "_frames"):
            return
        for i in range(0, self.n_frames(), max(1, int(step))):
            yield self._frames[i]
