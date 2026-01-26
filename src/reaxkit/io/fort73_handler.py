"""handler for parsing and cleaning data in fort.73 / energylog / fort.58 files"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from reaxkit.io.file_handler import FileHandler


class Fort73Handler(FileHandler):
    """
    Handler for ReaxFF energy-vs-iteration logs.

    Supported filenames (same logical table format):
      - fort.73
      - energylog
      - fort.58  (may wrap/continue values onto the next line)

    Behavior:
      - Parses header starting with 'Iter.'
      - Accumulates tokens across continuation lines
      - If a row has fewer values than headers (e.g. missing Eelec), pads with NaN
      - If a row wraps (e.g. last value printed on next line), appends continuation tokens
    """

    def __init__(self, file_path: str | Path = "fort.73"):
        super().__init__(file_path)

    @staticmethod
    def _is_int_token(tok: str) -> bool:
        # Iteration index appears as an integer token
        # (allow leading +/-, though usually non-negative)
        if not tok:
            return False
        if tok[0] in "+-":
            tok = tok[1:]
        return tok.isdigit()

    def _parse(self) -> tuple[pd.DataFrame, Dict[str, Any]]:
        cols: List[str] = []
        rows: List[List[Optional[str]]] = []

        current: List[Optional[str]] = []
        in_table = False

        def _flush_current() -> None:
            nonlocal current
            if not cols or not current:
                current = []
                return
            # pad missing trailing values (energylog case)
            if len(current) < len(cols):
                current = current + [None] * (len(cols) - len(current))
            # truncate any accidental extra tokens
            rows.append(current[: len(cols)])
            current = []

        with open(self.path, "r") as fh:
            for line in fh:
                s = line.strip()

                # skip blanks/separators
                if not s or "----" in s:
                    continue

                # header
                if s.startswith("Iter."):
                    cols = s.split()
                    in_table = True
                    current = []
                    continue

                if not in_table:
                    continue

                parts = s.split()
                if not parts:
                    continue

                # New row starts when first token is an integer iteration index.
                if self._is_int_token(parts[0]):
                    # finish previous row (even if incomplete)
                    _flush_current()
                    current = parts[:]  # start new row buffer
                else:
                    # continuation line (fort.58 style): append to current row
                    if not current:
                        # orphan continuation; ignore
                        continue
                    current.extend(parts)

                # If we already have a full row, flush it (handles cases where
                # continuation makes it complete before next Iter appears).
                if cols and len(current) >= len(cols):
                    _flush_current()

        # flush last buffered row at EOF
        _flush_current()

        df = pd.DataFrame(rows, columns=cols)

        # Convert numeric columns safely (None/garbage -> NaN)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if "Iter." in df.columns:
            df.rename(columns={"Iter.": "iter"}, inplace=True)

        meta: Dict[str, Any] = {
            "n_records": len(df),
            "columns": df.columns.tolist(),
            # optional debugging hints:
            "source_file": str(self.path),
        }

        self._frames = []  # Not used in this handler
        return df, meta
