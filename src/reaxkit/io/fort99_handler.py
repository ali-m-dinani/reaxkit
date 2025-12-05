"""handler for parsing and cleaning data in fort.99 file"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import re
import pandas as pd

from reaxkit.io.file_handler import FileHandler


class Fort99Handler(FileHandler):
    """
    Handler for fort.99 (ReaxFF training set error report).

    DataFrame columns:
        section       – CHARGE / HEATFO / GEOMETRY / CELL PARAMETERS / ENERGY / None
        title         – text descriptor ("Charge atom: 1", "Energy +butbenz/1 ...")
        ffield_value  – FF value
        qm_value      – QM/Lit value
        weight        – training weight
        error         – error reported in fort.99
        total_error   – cumulative weighted error
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
