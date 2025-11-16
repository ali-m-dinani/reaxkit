"""handler for parsing and cleaning data in fort.79 file"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import re
import math
import pandas as pd

from reaxkit.io.file_handler import FileHandler

# ----------------------------------------------------------------------
# Regex that matches:
#  - Proper Fortran float with optional D/E exponent:  1.234D+05, 3.21e-3, 0.5
#  - Malformed bare-exponent tokens we want to capture then mark as NaN: 0.2408814586-316
#    (We capture them so field counts stay correct; _f() will convert these to NaN.)
# ----------------------------------------------------------------------
_FNUM = r"[+-]?\d+\.\d+(?:[DdEe][+-]?\d+|[+-]\d+)?"
_FVAL_RE = re.compile(_FNUM)

def _f(s: str) -> float:
    """
    Convert a numeric token to float.
    - Valid Fortran floats 'D'/'d' → 'E'
    - Bare-exponent malformed tokens (e.g., '0.24088-316') → NaN
    - Any conversion error → NaN
    """
    try:
        clean = s.strip()
        # Bare exponent without D/E ('0.240...-316') → NaN by policy
        if re.fullmatch(r"[+-]?\d+\.\d+[+-]\d+", clean):
            return float("nan")
        # Normal path: Fortran 'D' → 'E'
        clean = clean.replace("D", "E").replace("d", "E")
        return float(clean)
    except Exception:
        return float("nan")


class Fort79Handler(FileHandler):
    """
    Handler for fort.79 files.

    Output DataFrame columns:
      ['identifier','value1','value2','value3',
       'diff1','diff2','diff3',
       'a','b','c',
       'parabol_min','parabol_min_diff',
       'value4','diff4']
    """

    def __init__(self, file_path: str | Path = "fort.79"):
        super().__init__(file_path)
        self._frames: List[pd.DataFrame] = []
        self._n_records: Optional[int] = None

    def _parse(self) -> tuple[pd.DataFrame, Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        with open(self.path, "r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()

        i, n = 0, len(lines)
        while i < n:
            line = lines[i]
            if line.strip().startswith("Values used for parameter"):
                ident = line.split("parameter", 1)[1].strip()

                # ---- three "Values used..." numbers (may wrap) ----
                v1 = v2 = v3 = math.nan
                i += 1
                if i < n:
                    vals = _FVAL_RE.findall(lines[i])
                    if len(vals) < 3 and i + 1 < n:
                        vals += _FVAL_RE.findall(lines[i + 1])
                        # only advance extra line if we actually used it
                        i += 1 if len(vals) >= 3 else 0
                    if len(vals) >= 3:
                        v1, v2, v3 = map(_f, vals[:3])

                # ---- "Differences found" + three diffs (may wrap) ----
                d1 = d2 = d3 = math.nan
                i += 1
                if i < n and lines[i].strip().startswith("Differences found"):
                    i += 1
                if i < n:
                    diffs = _FVAL_RE.findall(lines[i])
                    if len(diffs) < 3 and i + 1 < n:
                        diffs += _FVAL_RE.findall(lines[i + 1])
                        i += 1 if len(diffs) >= 3 else 0
                    if len(diffs) >= 3:
                        d1, d2, d3 = map(_f, diffs[:3])

                # ---- "Parabol: a= ... b= ... c= ..."  (may wrap) ----
                a = b = c = math.nan
                i += 1
                if i < n:
                    par_chunk = lines[i]
                    # try to pull from next lines if needed
                    if i + 1 < n:
                        par_chunk_try = par_chunk + " " + lines[i + 1]
                    else:
                        par_chunk_try = par_chunk
                    if i + 2 < n:
                        par_chunk_try2 = par_chunk_try + " " + lines[i + 2]
                    else:
                        par_chunk_try2 = par_chunk_try

                    # Try 1-line, 2-line, 3-line match
                    m = re.search(r"a=\s*(" + _FNUM + r")\s*b=\s*(" + _FNUM + r")\s*c=\s*(" + _FNUM + r")", par_chunk)
                    bump = 0
                    if not m:
                        m = re.search(r"a=\s*(" + _FNUM + r")\s*b=\s*(" + _FNUM + r")\s*c=\s*(" + _FNUM + r")", par_chunk_try)
                        bump = 1 if m else 0
                    if not m:
                        m = re.search(r"a=\s*(" + _FNUM + r")\s*b=\s*(" + _FNUM + r")\s*c=\s*(" + _FNUM + r")", par_chunk_try2)
                        bump = 2 if m else 0
                    if m:
                        a, b, c = map(_f, m.groups())
                        i += bump  # consume extra lines used

                # ---- "Minimum of the parabol ..." ----
                parabol_min = math.nan
                i += 1
                if i < n:
                    mins = _FVAL_RE.findall(lines[i])
                    if mins:
                        parabol_min = _f(mins[0])

                # ---- "Difference belonging to minimum of parabol ..." ----
                parabol_min_diff = math.nan
                i += 1
                if i < n:
                    mins2 = _FVAL_RE.findall(lines[i])
                    if mins2:
                        parabol_min_diff = _f(mins2[0])

                # ---- "New parameter value ..." ----
                value4 = math.nan
                i += 1
                if i < n:
                    news = _FVAL_RE.findall(lines[i])
                    if news:
                        value4 = _f(news[0])

                # ---- "Difference belonging to new parameter value ..." ----
                diff4 = math.nan
                i += 1
                if i < n:
                    news2 = _FVAL_RE.findall(lines[i])
                    if news2:
                        diff4 = _f(news2[0])

                rows.append(
                    {
                        "identifier": ident,
                        "value1": v1, "value2": v2, "value3": v3,
                        "diff1": d1, "diff2": d2, "diff3": d3,
                        "a": a, "b": b, "c": c,
                        "parabol_min": parabol_min,
                        "parabol_min_diff": parabol_min_diff,
                        "value4": value4,
                        "diff4": diff4,
                    }
                )
            i += 1

        df = pd.DataFrame(
            rows,
            columns=[
                "identifier",
                "value1", "value2", "value3",
                "diff1", "diff2", "diff3",
                "a", "b", "c",
                "parabol_min", "parabol_min_diff",
                "value4", "diff4",
            ],
        )

        meta: Dict[str, Any] = {"n_records": int(len(df))}
        self._frames = []
        return df, meta

    def n_frames(self) -> int:
        return 0

    def iter_frames(self, step: int = 1) -> Iterator[Dict[str, Any]]:
        if False:
            yield {}
