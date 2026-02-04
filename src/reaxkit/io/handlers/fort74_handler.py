"""
ReaxFF structure summary (fort.74) handler.

This module provides a handler for parsing ReaxFF ``fort.74`` files,
which report per-structure energetic and thermodynamic summary
quantities produced during ReaxFF runs or force-field training.

Typical use cases include:

- extracting formation energies and volumes
- comparing multiple structures or configurations
- building datasets for bulk-property analysis
"""


from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from reaxkit.io.base_handler import FileHandler


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
    Parser for ReaxFF structure summary files (``fort.74``).

    This class parses ``fort.74`` files and exposes one summary record
    per structure or configuration as a tabular dataset.

    Parsed Data
    -----------
    Summary table
        One row per structure, returned by ``dataframe()``, with columns:
        ["identifier", "Emin", "iter", "Hf", "V", "D"]

        All columns except ``identifier`` are optional and may contain
        ``NaN`` when the corresponding quantity is not present in the file.

    Metadata
        Returned by ``metadata()``, containing:
        ["n_records", "n_frames"]

    Notes
    -----
    - The ``identifier`` is taken from the first token of each line.
    - Field labels (e.g. ``Emin:``, ``Iter.:``, ``Hf:``, ``Vol:``, ``Dens:``)
      are detected dynamically and may appear in any order.
    - This handler is not frame-based; ``n_frames()`` always returns 0.
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
