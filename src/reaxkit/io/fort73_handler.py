"""handler for parsing and cleaning data in fort.73 file"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any

from reaxkit.io.file_handler import FileHandler


class Fort73Handler(FileHandler):
    """
    Handler for ReaxFF energy-vs-iteration logs.

    Supported filenames:
      - fort.73
      - energylog

    - Reads energy terms vs iteration
    - Parses into a clean DataFrame (no analysis here)
    """

    def __init__(self, file_path: str | Path = "fort.73"):
        super().__init__(file_path)

    def _parse(self) -> tuple[pd.DataFrame, Dict[str, Any]]:
        rows: List[list] = []
        cols: List[str] = []

        with open(self.path, "r") as fh:
            for line in fh:
                # Detect header line
                if line.strip().startswith("Iter."):
                    cols = line.strip().split()
                    continue

                # Skip non-data lines
                if not line.strip() or "----" in line:
                    continue

                parts = line.strip().split()
                if len(parts) < len(cols):
                    parts = parts + [None] * (len(cols) - len(parts))
                rows.append(parts[:len(cols)])

        # Build DataFrame
        df = pd.DataFrame(rows, columns=cols)

        # Convert numeric columns
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Rename only Iter. → iter
        if "Iter." in df.columns:
            df.rename(columns={"Iter.": "iter"}, inplace=True)

        meta: Dict[str, Any] = {
            "n_records": len(df),
            "columns": df.columns.tolist(),
        }

        self._frames = []  # Not used in fort.73
        return df, meta
