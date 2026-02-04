"""
ReaxFF electric-field output (fort.78) handler.

This module provides a handler for parsing ReaxFF ``fort.78`` files,
which report per-iteration electric-field components and magnitudes
during simulations with applied external fields.

Typical use cases include:

- analyzing applied electric-field schedules
- correlating field strength with polarization or dipole response
- plotting field components versus iteration
"""


from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import pandas as pd

from reaxkit.io.base_handler import BaseHandler

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


class Fort78Handler(BaseHandler):
    """
    Parser for ReaxFF electric-field output files (``fort.78``).

    This class parses ``fort.78`` files and exposes electric-field
    quantities as a tidy, iteration-indexed table with canonical
    column names.

    Parsed Data
    -----------
    Summary table
        One row per iteration, returned by ``dataframe()``, with columns:

        - When 5 columns are present:
          ["iter", "field_x", "field_y", "field_z", "E_field"]

        - When 8 columns are present:
          ["iter", "field_x", "field_y", "field_z",
           "E_field_x", "E_field_y", "E_field_z", "E_field"]

        - For other column counts:
          ["Col1", "Col2", ..., "ColN"]

    Metadata
        Returned by ``metadata()``, containing:
        ["source", "n_rows", "has_time", "columns"]

    Notes
    -----
    - Header presence is detected automatically.
    - Canonical column names are enforced when column counts match
      known ``fort.78`` formats.
    - Duplicate iteration indices are resolved by keeping the last entry.
    - This handler represents a scalar-per-iteration time-series file.
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
