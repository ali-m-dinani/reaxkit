# reaxkit/io/fort76_handler.py

"""
Handler for fort.76 files (restraint monitor table).

Expected columns per row (whitespace-delimited):
  iter  E_res  E_pot  r1_target  r1_actual  r2_target  r2_actual  ...

- Supports an arbitrary number of restraints (pairs of target/actual).
- Skips blank/comment/header lines robustly.
- Drops duplicate iterations (keeps last).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

from reaxkit.io.file_handler import FileHandler


class Fort76Handler(FileHandler):
    """
    Handler for fort.76 files.
    - Reads raw file, parses into a summary DataFrame.
    - Does only parsing/cleaning; no analysis or plotting.
    """

    def __init__(self, file_path: str | Path = "fort.76"):
        super().__init__(file_path)
        self._frames: List[pd.DataFrame] = []  # optional, kept for template consistency
        self._n_records: Optional[int] = None

    @staticmethod
    def _is_float(token: str) -> bool:
        try:
            float(token)
            return True
        except Exception:
            return False

    def _parse(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        rows: List[List[float]] = []
        n_restraints_max = 0

        with open(self.path, "r") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                # Common comment styles
                if s.startswith(("#", "!", "//")):
                    continue

                parts = s.split()
                if len(parts) < 3:
                    continue

                # If the line doesn't look numeric, treat as header and skip
                if not (self._is_float(parts[0]) and self._is_float(parts[1]) and self._is_float(parts[2])):
                    continue

                # Convert tokens to floats (iter will be cast to int later)
                try:
                    vals = [float(x) for x in parts]
                except Exception:
                    continue

                # Determine restraint pairs beyond (iter, E_res, E_pot)
                extra = len(vals) - 3
                if extra < 0:
                    continue

                # If odd number of extra columns, ignore this malformed row
                if extra % 2 != 0:
                    continue

                n_restraints = extra // 2
                n_restraints_max = max(n_restraints_max, n_restraints)

                rows.append(vals)

        # If nothing parsed, return empty but well-defined DF
        base_cols = ["iter", "E_res", "E_pot"]
        if not rows:
            df_empty = pd.DataFrame(columns=base_cols)
            meta = {
                "n_records": 0,
                "n_frames": 0,
                "n_restraints": 0,
                "restraint_cols": [],
            }
            self._frames = []
            return df_empty, meta

        # Pad rows so all have same number of restraint columns (in case file changes mid-run)
        target_len = 3 + 2 * n_restraints_max
        for r in rows:
            if len(r) < target_len:
                r.extend([float("nan")] * (target_len - len(r)))

        # Build columns dynamically
        cols = list(base_cols)
        for i in range(1, n_restraints_max + 1):
            cols.append(f"r{i}_target")
            cols.append(f"r{i}_actual")

        df = pd.DataFrame(rows, columns=cols)

        # Types
        df["iter"] = df["iter"].astype(int, errors="ignore")

        # Clean: drop duplicate iterations (keep last)
        if not df.empty:
            keep_idx = df.drop_duplicates("iter", keep="last").index
            df = df.loc[keep_idx].reset_index(drop=True)

        self._frames = []  # fort.76 is already 1-row-per-iter; no extra per-frame tables needed

        meta: Dict[str, Any] = {
            "n_records": int(len(df)),
            "n_frames": int(len(df)),
            "n_restraints": int(n_restraints_max),
            "restraint_cols": [c for c in df.columns if c.startswith("r")],
        }
        return df, meta

    # ---- File-specific accessors
    def n_frames(self) -> int:
        # 1 row per iteration
        return int(self.metadata().get("n_frames", len(self.dataframe())))

    def n_restraints(self) -> int:
        return int(self.metadata().get("n_restraints", 0))

    def frame(self, i: int) -> Dict[str, Any]:
        """
        Return a normalized per-row structure.
        restraints is a list of dicts:
          [{"index": 1, "target": ..., "actual": ...}, ...]
        """
        df = self.dataframe()
        row = df.iloc[i]

        restraints: List[Dict[str, Any]] = []
        n = self.n_restraints()
        for k in range(1, n + 1):
            tgt = row.get(f"r{k}_target")
            act = row.get(f"r{k}_actual")
            # Skip if both are NaN (e.g., padded)
            if pd.isna(tgt) and pd.isna(act):
                continue
            restraints.append({"index": k, "target": tgt, "actual": act})

        return {
            "index": int(i),
            "iter": int(row.get("iter")),
            "E_res": row.get("E_res"),
            "E_pot": row.get("E_pot"),
            "restraints": restraints,
        }

    def iter_frames(self, step: int = 1) -> Iterator[Dict[str, Any]]:
        for i in range(0, self.n_frames(), max(1, int(step))):
            yield self.frame(i)
