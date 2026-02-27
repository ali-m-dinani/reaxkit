"""
ReaxFF molecular fragment analysis (molfra.out) handler.

This module provides a handler for parsing ReaxFF ``molfra.out`` and
``molfra_ig.out`` files, which report molecule/fragment compositions
and their frequencies as a function of simulation iteration.

Typical use cases include:

- tracking molecular species formation and decay
- monitoring reaction pathways and fragment distributions
- computing molecule counts and system-level mass summaries
"""


from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Iterator, Optional
import pandas as pd

from reaxkit.engine.reaxff.io.base import BaseHandler


class MolFraHandler(BaseHandler):
    """
    Parser for ReaxFF molecular fragment output files
    (``molfra.out``, ``molfra_ig.out``).

    This class parses molecular fragment frequency data and exposes both
    per-molecule and per-iteration summary information as structured
    tabular datasets.

    Parsed Data
    -----------
    Molecule table
        One row per (iteration, molecular species), returned by
        ``dataframe()``, with columns:
        ["iter", "molecular_formula", "freq", "molecular_mass"]

    Totals table
        One row per iteration, accessible via ``totals()``, with columns:
        ["iter", "total_molecules", "total_atoms", "total_molecular_mass"]

    Notes
    -----
    - Molecular species are identified by their chemical formula strings.
    - Frequency values represent counts per iteration.
    - Totals are parsed from summary blocks following molecule listings.
    - This handler is iteration-based rather than frame-based, but exposes
      a minimal frame-like API for consistency.
    """
    def __init__(self, file_path: str | Path = "molfra.out", reporter=None):
        """
        Initialize the instance.

        Parameters
        ----------
        file_path : str | Path
            Parameter description.

        """
        super().__init__(file_path)
        self._n_records: Optional[int] = None
        self._types: Optional[List[str]] = None
        self._iters: list[int] = []
        self._reporter = reporter

    # ---- Core parser
    def _parse(self) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Parse molfra.out into two dataframes:
          1. Molecule occurrences per iter
          2. Totals (number of molecules, atoms, system molecular_mass) per iter
        """
        mol_rows: list[dict[str, Any]] = []
        total_rows: list[dict[str, Any]] = []
        current_iter: Optional[int] = None
        current_totals: dict[str, Any] = {}

        with open(self.path, "r") as fh:
            raw_lines = fh.readlines()

        total_lines = len(raw_lines)
        for line_i, line in enumerate(raw_lines, start=1):
            line = line.strip()
            if self._reporter:
                self._reporter("load", line_i, total_lines, "Parsing molfra file")
            if not line or line.startswith("Bond order"):
                continue

            if line.startswith("Iteration"):
                continue

            if line.startswith("Total number of molecules"):
                current_totals["total_molecules"] = int(line.split()[-1])
                continue
            if line.startswith("Total number of atoms"):
                current_totals["total_atoms"] = int(line.split()[-1])
                continue
            if line.startswith("Total system"):
                current_totals["total_molecular_mass"] = float(line.split()[-1])
                if current_iter is not None and current_totals:
                    current_totals["iter"] = current_iter
                    total_rows.append(current_totals)
                    current_totals = {}
                continue

            parts = line.split()
            if len(parts) >= 5 and "x" in parts:
                try:
                    iter_val = int(parts[0])
                    freq = int(parts[1])
                    molecular_mass = float(parts[-1])
                    x_index = parts.index("x")
                    molecular_formula = parts[x_index + 1]
                    current_iter = iter_val
                except (ValueError, IndexError):
                    continue
                mol_rows.append({
                    "iter": iter_val,
                    "molecular_formula": molecular_formula,
                    "freq": freq,
                    "molecular_mass": molecular_mass,
                })

        # Build dataframes
        df_mol = pd.DataFrame(mol_rows)
        df_tot = pd.DataFrame(total_rows).sort_values("iter").reset_index(drop=True)

        # Store parsed tables for convenience accessors.
        self._df_totals = df_tot
        self._n_records = len(df_mol)
        self._types = sorted(df_mol["molecular_formula"].unique().tolist()) if not df_mol.empty else []
        self._iters = sorted(df_mol["iter"].unique().tolist()) if not df_mol.empty else []
        return df_mol, {}

    # ---- Convenience accessors (file-specific)
    def n_records(self) -> int:
        """
        N records.

        Returns
        -------
        int
            Return value description.

        """
        if self._n_records is None:
            self.parse()
        return int(self._n_records or 0)

    def molecular_formulas(self) -> List[str]:
        """
        Molecular formulas.

        Returns
        -------
        List[str]
            Return value description.

        """
        if self._types is None:
            self.parse()
        return list(self._types or [])

    def by_type(self, mtype: str) -> pd.DataFrame:
        """
        By type.

        Parameters
        ----------
        mtype : str
            Parameter description.

        Returns
        -------
        pd.DataFrame
            Return value description.

        """
        df = self.dataframe()
        return df[df["molecular_formula"] == mtype].reset_index(drop=True)

    def totals(self) -> pd.DataFrame:
        """
        Totals.

        Returns
        -------
        pd.DataFrame
            Return value description.

        """
        if hasattr(self, "_df_totals"):
            return self._df_totals.copy()
        else:
            raise AttributeError("Totals dataframe not parsed or unavailable.")

    # ---- Frame-oriented API (kept minimal for template parity)
    def n_frames(self) -> int:
        """
        N frames.

        Returns
        -------
        int
            Return value description.

        """
        if not self._iters:
            self.parse()
        return len(self._iters)

    def frame(self, i: int) -> Dict[str, Any]:
        """
        Return a per-iter 'frame' view:
          { 'iter': <int>, 'freqs': DataFrame[molecular_formula, freq] }
        """
        df = self.dataframe()
        if df.empty:
            raise IndexError("No data loaded.")
        iters = sorted(df["iter"].unique())
        if i < 0 or i >= len(iters):
            raise IndexError(f"Frame index {i} out of range (0..{len(iters)-1}).")
        it = iters[i]
        sub = (
            df.loc[df["iter"] == it, ["molecular_formula", "freq"]]
              .sort_values("molecular_formula")
              .reset_index(drop=True)
        )
        return {"iter": it, "freqs": sub}

    def iter_frames(self, step: int = 1) -> Iterator[Dict[str, Any]]:
        """
        Iter frames.

        Parameters
        ----------
        step : int
            Parameter description.

        Yields
        -------
        Iterator[Dict[str, Any]]
            Return value description.

        """
        for i in range(0, self.n_frames(), step):
            yield self.frame(i)
