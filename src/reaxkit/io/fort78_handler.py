"""handler for parsing and cleaning data in fort.78 file"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import pandas as pd

from reaxkit.io.file_handler import FileHandler

# Canonical names as requested
_CANONICAL_5 = [
    "iter",
    "field_x", "field_y", "field_z",
    "E_field",
]

_CANONICAL_8 = [
    "iter",
    "field_x", "field_y", "field_z",
    "E_field_x", "E_field_y", "E_field_z",
    "E_field",
]

_NUMERIC_CANONICAL = set(_CANONICAL_8)  # superset of _CANONICAL_5


class Fort78Handler(FileHandler):
    """
    Parse ReaxFF fort.78 into a tidy per-iteration table.

    Behavior:
      - Regardless of header presence, columns are renamed to canonical lists:
          * 5 cols  -> _CANONICAL_5
          * 8 cols  -> _CANONICAL_8
      - Unknown column counts are left as Col1..ColN (but still parsed).
    """

    def __init__(self, file_path: str | Path = "fort.78"):
        super().__init__(file_path)
        self._n_rows: int = 0

    def _parse(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        path = Path(self.path)

        # Peek first line to decide if it's a header or data
        with open(path, "r") as fh:
            first_line = fh.readline().strip()
        first_tokens = first_line.split()
        is_numeric_row = all(self._is_number(tok) for tok in first_tokens)

        # Read file using whitespace separator (no deprecation warning)
        if is_numeric_row:
            # No header present
            df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
        else:
            # Header present -> parse with header row, then overwrite with canonical names
            df = pd.read_csv(path, sep=r"\s+", header=0, engine="python")

        ncols = df.shape[1]

        # Force canonical names when column count matches 5 or 8
        if ncols == 5:
            df.columns = _CANONICAL_5
        elif ncols == 8:
            df.columns = _CANONICAL_8
        else:
            # Fallback: numbered names (still parsed)
            df.columns = [f"Col{i+1}" for i in range(ncols)]

        # Coerce numeric for known numeric columns that are present
        for col in _NUMERIC_CANONICAL:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Clean 'iter' column if present
        if "iter" in df.columns:
            df = df.dropna(subset=["iter"])
            df["iter"] = pd.to_numeric(df["iter"], errors="coerce")
            df = df.dropna(subset=["iter"])
            try:
                df["iter"] = df["iter"].astype(int)
            except Exception:
                pass
            df = df.drop_duplicates("iter", keep="last").reset_index(drop=True)

        self._n_rows = len(df)
        meta: Dict[str, Any] = {
            "source": "fort.78",
            "n_rows": self._n_rows,
            "has_time": False,
            "columns": list(df.columns),
        }
        return df, meta

    def n_rows(self) -> int:
        return self._n_rows

    @staticmethod
    def _is_number(tok: str) -> bool:
        try:
            float(tok.replace("D", "E").replace("d", "E"))
            return True
        except ValueError:
            return False
