"""
ReaxFF training set error report (fort.99) handler.

This module provides a handler for parsing ReaxFF ``fort.99`` files,
which summarize force-field training errors by category and target
during parameter optimization runs.

Typical use cases include:

- analyzing training set contributions to total error
- inspecting charge, geometry, and energy fitting quality
- building diagnostics for force-field parameterization workflows
"""


from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import re
import pandas as pd

from reaxkit.io.base_handler import FileHandler


class Fort99Handler(FileHandler):
    """
    Parser for ReaxFF training set error reports (``fort.99``).

    This class parses ``fort.99`` files and exposes individual training
    targets and their contributions to the total force-field error as
    a structured tabular dataset.

    Parsed Data
    -----------
    Summary table
        One row per training target, returned by ``dataframe()``,
        with columns:
        ["lineno", "section", "title",
         "ffield_value", "qm_value", "weight",
         "error", "total_ff_error"]

        The ``section`` column categorizes each target as one of:
        ["CHARGE", "HEATFO", "GEOMETRY", "CELL PARAMETERS", "ENERGY", None].

    Metadata
        Returned by ``metadata()``, containing:
        ["n_records", "n_frames"]

    Notes
    -----
    - The last five numeric values on each line are interpreted as
      (FF value, QM/reference value, weight, error, total error).
    - Section categories are inferred heuristically from the title text.
    - Unrecognized entries are retained with ``section=None``.
    - This handler is not frame-based; ``n_frames()`` always returns 0.
    """

    def __init__(self, file_path: str | Path = "fort.99"):
        super().__init__(file_path)

    def _parse(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        # float like -17.8000, 1.54, 1.0e-03, etc.
        float_re = re.compile(r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?")

        with open(self.path, "r") as fh:
            for lineno, raw in enumerate(fh, start=1):
                line = raw.rstrip("\n")
                if not line.strip():
                    continue

                # Find all floats
                matches = list(float_re.finditer(line))
                if len(matches) < 5:
                    continue

                # Extract last 5 numbers
                last5 = matches[-5:]
                vals = [float(m.group()) for m in last5]
                ffield_val, qm_val, weight, err, tot_err = vals

                # Title
                title_start = last5[0].start()
                title = line[:title_start].strip()
                if not title:
                    continue

                tl = title.lower()

                # -------- SECTION detection --------
                if "charge" in tl:
                    section = "CHARGE"
                elif "heat" in tl:
                    section = "HEATFO"
                elif ("bond" in tl) or ("angle" in tl):
                    section = "GEOMETRY"
                elif ("a:" in tl) or ("b:" in tl) or ("c:" in tl):
                    section = "CELL PARAMETERS"
                elif "energy" in tl:
                    section = "ENERGY"
                else:
                    section = None  # mark unknown
                    print(f"Unrecognized fort.99 entry at line {lineno}: {title}")

                # Save row
                rows.append(
                    {
                        "lineno": lineno,
                        "section": section,
                        "title": title,
                        "ffield_value": ffield_val,
                        "qm_value": qm_val,
                        "weight": weight,
                        "error": err,
                        "total_ff_error": tot_err,
                    }
                )

        df = pd.DataFrame(rows)

        # fort.99 has no per-frame data
        self._frames = []  # from TemplateHandler
        meta: Dict[str, Any] = {
            "n_records": len(df),
            "n_frames": 0,
        }
        return df, meta
