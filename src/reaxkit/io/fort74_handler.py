from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re
import pandas as pd

from reaxkit.io.template_handler import TemplateHandler


class Fort74Handler(TemplateHandler):
    """
    Handler for fort.74 files.

    Expected line format (no strict header required):

        bulk_c5             Emin:     -180.170 Iter.:     0 Hf:    221.861 Vol:     26.731 Dens):     16.180

    Parsed columns:
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

                parts = line.split()

                # Format:
                # 0: identifier
                # 1: Emin:
                # 2: Emin_value
                # 3: Iter.:
                # 4: Iter_value
                # 5: Hf:
                # 6: Hf_value
                # 7: Vol:
                # 8: Vol_value
                # 9: Dens) or Dens:
                # 10: Dens_value

                try:
                    identifier = parts[0]
                    Emin = float(parts[2])
                    Iter = int(float(parts[4]))
                    Hf = float(parts[6])
                    Vol = float(parts[8])
                    Dens = float(parts[10])
                except Exception:
                    # If parsing fails → skip line
                    continue

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

    # Optional: override frame-related methods to avoid confusion
    def n_frames(self) -> int:
        return 0

    def frame(self, i: int) -> Dict[str, Any]:
        raise IndexError("fort.74 has no per-frame data")

