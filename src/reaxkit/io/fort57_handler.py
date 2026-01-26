# reaxkit/io/fort57_handler.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from reaxkit.io.file_handler import FileHandler


class Fort57Handler(FileHandler):
    """
    Handler for ReaxFF fort.57 files.

    Expected format (example):
        <geo_descriptor>
          Iter.   Epot   Temp(K)   Tset(K)   RMSG   nfc
             0    ...
             1    ...

    Notes
    -----
    - First line is treated as metadata: geo_descriptor (string).
    - Remaining lines are parsed as a numeric table with columns:
        iter, E_pot, T, T_set, RMSG, nfc
    - Duplicate iterations are dropped (keep last), similar to other handlers.
    """

    _COLS = ["iter", "E_pot", "T", "T_set", "RMSG", "nfc"]

    def __init__(self, file_path: str | Path = "fort.57"):
        super().__init__(file_path)
        self._geo_descriptor: str = ""

    def _parse(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        with open(self.path, "r") as fh:
            lines = fh.readlines()

        if not lines:
            df = pd.DataFrame(columns=self._COLS)
            meta: Dict[str, Any] = {"geo_descriptor": "", "n_records": 0}
            return df, meta

        # 1) metadata line
        self._geo_descriptor = lines[0].strip()

        # 2) parse numeric rows
        rows: List[List[Any]] = []
        for raw in lines[1:]:
            s = raw.strip()
            if not s:
                continue

            toks = s.split()

            # Skip headers / non-data lines (e.g., "Iter.", "Epot", etc.)
            # Data rows should have 6 tokens and start with an integer iter.
            if len(toks) < 6:
                continue
            if not toks[0].lstrip("+-").isdigit():
                continue

            try:
                it = int(toks[0])
                epot = float(toks[1])
                temp = float(toks[2])
                tset = float(toks[3])
                rmsg = float(toks[4])
                nfc = int(float(toks[5]))  # sometimes written like "-1" but be tolerant
            except Exception:
                continue

            rows.append([it, epot, temp, tset, rmsg, nfc])

        df = pd.DataFrame(rows, columns=self._COLS)

        # 3) clean: drop duplicate iters (keep last)
        if not df.empty:
            keep_idx = df.drop_duplicates("iter", keep="last").index
            df = df.loc[keep_idx].sort_values("iter").reset_index(drop=True)

        meta = {
            "geo_descriptor": self._geo_descriptor,
            "n_records": int(len(df)),
            "n_frames": int(len(df)),  # for consistency with other handlers
        }
        return df, meta

    # ---- convenience accessors ----
    @property
    def geo_descriptor(self) -> str:
        """First-line metadata string (e.g., geometry descriptor)."""
        return self._geo_descriptor or str(self.metadata().get("geo_descriptor", ""))

    def n_frames(self) -> int:
        return int(self.metadata().get("n_frames", 0))
