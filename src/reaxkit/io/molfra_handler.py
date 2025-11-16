"""handler for parsing and cleaning data in molfra.out or molfra_ig.out file"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Iterator, Optional
import pandas as pd

from reaxkit.io.file_handler import FileHandler


class MolFraHandler(FileHandler):
    """
    Handler for `molfra.out` — parses ReaxFF molecule/fragment frequencies by iter.

    Outputs
    -------
    - DataFrame (df_mol): per-iter molecule data with columns:
        ['iter', 'molecular_formula', 'freq', 'molecular_mass']
    - Metadata (meta): dict summarizing n_records, n_iters, iter range, and molecule types
    - DataFrame (df_totals): totals per iter including total_molecules, total_atoms, total_molecular_mass

    Notes
    -----
    Parses only and does not perform analysis or plotting.
    """


    def __init__(self, file_path: str | Path = "molfra.out"):
        super().__init__(file_path)
        self._n_records: Optional[int] = None
        self._types: Optional[List[str]] = None

    # ---- Core parser
    def _parse(self) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Parse molfra.out into two dataframes:
          1. Molecule occurrences per iter
          2. Totals (number of molecules, atoms, system molecular_mass) per iter
        """
        mol_rows: list[dict[str, any]] = []
        total_rows: list[dict[str, any]] = []

        with open(self.path, "r") as fh:
            current_iter = None
            current_totals = {}

            for line in fh:
                line = line.strip()
                if not line or line.startswith("Bond order"):
                    continue

                # Header
                if line.startswith("Iteration"):
                    continue

                # Total lines
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

                # Molecule lines
                parts = line.split()
                if len(parts) >= 5 and "x" in parts:
                    try:
                        iter = int(parts[0])
                        freq = int(parts[1])
                        molecular_mass = float(parts[-1])
                        x_index = parts.index("x")
                        molecular_formula = parts[x_index + 1]
                        current_iter = iter
                    except (ValueError, IndexError):
                        continue
                    mol_rows.append({
                        "iter": iter,
                        "molecular_formula": molecular_formula,
                        "freq": freq,
                        "molecular_mass": molecular_mass,
                    })

        # Build dataframes
        df_mol = pd.DataFrame(mol_rows)
        df_tot = pd.DataFrame(total_rows).sort_values("iter").reset_index(drop=True)

        meta = {
            "n_records": len(df_mol),
            "n_iters": df_mol["iter"].nunique(),
            "iter_min": df_mol["iter"].min(),
            "iter_max": df_mol["iter"].max(),
            "molecular_formulas": sorted(df_mol["molecular_formula"].unique().tolist()),
        }

        # Store both
        self._df_totals = df_tot
        self._n_records = meta["n_records"]
        self._types = meta["molecular_formulas"]
        return df_mol, meta

    # ---- Convenience accessors (file-specific)
    def n_records(self) -> int:
        return int(self.metadata().get("n_records", 0))

    def molecular_formulas(self) -> List[str]:
        return list(self.metadata().get("molecular_formulas", []))

    def by_type(self, mtype: str) -> pd.DataFrame:
        """Return rows for a given molecule type."""
        df = self.dataframe()
        return df[df["molecular_formula"] == mtype].reset_index(drop=True)

    def totals(self) -> pd.DataFrame:
        """Return total molecules/atoms/molecular_mass per iter."""
        if hasattr(self, "_df_totals"):
            return self._df_totals.copy()
        else:
            raise AttributeError("Totals dataframe not parsed or unavailable.")

    # ---- Frame-oriented API (kept minimal for template parity)
    def n_frames(self) -> int:
        """molfra.out is not frame-based; expose unique iters instead."""
        return int(self.metadata().get("n_iters", 0))

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
        for i in range(0, self.n_frames(), step):
            yield self.frame(i)
