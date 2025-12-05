"""handler for parsing and cleaning data in fort.74 file"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from reaxkit.io.file_handler import FileHandler


def _safe_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _safe_int(s: str) -> int | None:
    try:
        # sometimes written as "0", "0.0", etc.
        return int(float(s))
    except (ValueError, TypeError):
        return None


class Fort74Handler(FileHandler):
    """
    Handler for fort.74 files.

    Example line formats (fields may be missing):

        bulk_c5             Emin:     -180.170 Iter.:     0 Hf:    221.861 Vol:     26.731 Dens):     16.180
        Zn_nh3-1_P2         Emin:      103.888 Iter.:     0 Heatfo:    677.707
        N2                  Emin:     -236.672 Iter.:     0 Hf:      5.665 Vol:  15625.000 Dens):      0.003

    Parsed columns (all optional except identifier):
        identifier, Emin, iter, Hf, V, D
    """

    def __init__(self, file_path: str | Path = "fort.74"):
        super().__init__(file_path)

    def _parse(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        with open(self.path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue

                tokens = line.split()
                if not tokens:
                    continue

                # identifier is always the first token
                identifier = tokens[0]

                Emin = None
                Iter = None
                Hf = None
                Vol = None
                Dens = None

                # Walk through tokens and look for known labels like "Emin:", "Iter.:", etc.
                i = 1
                n = len(tokens)
                while i < n:
                    t = tokens[i]

                    # Emin:
                    if t == "Emin:" and i + 1 < n:
                        Emin = _safe_float(tokens[i + 1])
                        i += 2
                        continue

                    # Iter.: or Iter:
                    if (t == "Iter.:" or t == "Iter:") and i + 1 < n:
                        Iter = _safe_int(tokens[i + 1])
                        i += 2
                        continue

                    # Hf: or Heatfo:
                    if (t == "Hf:" or t == "Heatfo:") and i + 1 < n:
                        Hf = _safe_float(tokens[i + 1])
                        i += 2
                        continue

                    # Vol:
                    if t == "Vol:" and i + 1 < n:
                        Vol = _safe_float(tokens[i + 1])
                        i += 2
                        continue

                    # Dens: or Dens):
                    if (t == "Dens:" or t == "Dens):") and i + 1 < n:
                        Dens = _safe_float(tokens[i + 1])
                        i += 2
                        continue

                    # anything else → skip
                    i += 1

                rows.append(
                    {
                        "identifier": identifier,
                        "Emin": Emin,
                        "iter": Iter,
                        "Hf": Hf,
                        "V": Vol,
                        "D": Dens,
                    }
                )

        df = pd.DataFrame(rows, columns=["identifier", "Emin", "iter", "Hf", "V", "D"])

        self._frames = []
        meta = {"n_records": len(df), "n_frames": 0}

        return df, meta

    # fort.74 is effectively a single "table", not frame-based
    def n_frames(self) -> int:
        return 0

    def frame(self, i: int) -> Dict[str, Any]:
        raise IndexError("fort.74 has no per-frame data")
