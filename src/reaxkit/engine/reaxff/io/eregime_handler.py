"""
ReaxFF electric-field regime (eregime.in) handler.

This module provides a handler for parsing ReaxFF ``eregime.in`` files,
which define time-dependent electric-field schedules used in MD runs.

Typical use cases include:

- reading electric-field magnitudes and directions
- mapping field schedules to simulation iterations
- converting iteration indices to physical time

**Usage context**

- ReaxFF parsing: Read ReaxFF text outputs into normalized tabular structures.
- Workflow ingestion: Provide canonical handler interfaces used by adapters/workflows.
- Diagnostics/export: Preserve parsed metadata for reporting and downstream conversion.
"""


from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd

from reaxkit.engine.reaxff.io.base import BaseHandler


class EregimeHandler(BaseHandler):
    """
    Parser for ReaxFF electric-field schedule files (``eregime.in``).

    This class parses electric-field regime definitions and exposes them
    as structured tabular data suitable for downstream analysis and
    visualization.

    Parsed Data
    -----------
    Summary table
        One row per schedule entry, returned by ``dataframe()``, with columns:

        - If the maximum number of field zones is ≤ 1:
          ["iter", "field_zones", "field_dir", "field"]

        - If multiple field zones are present:
          ["iter", "field_zones",
           "field_dir1", "field1",
           "field_dir2", "field2", ...]

    Metadata
        Returned by ``metadata()``, containing:
        ["columns", "max_field_zones", "n_records"]

    Notes
    -----
    - Comment lines starting with ``#`` are ignored.
    - Missing direction/field pairs are padded with ``NaN`` to ensure a
      rectangular table.
    - Field directions are stored as strings; field magnitudes are numeric.
    """

    def __init__(self, file_path: str | Path = "eregime.in", reporter=None):
        """
        Initialize the instance.

        Parameters
        ----------
        file_path : str | Path
            Parameter description.

        """
        super().__init__(file_path)
        self._reporter = reporter

    def _parse(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
         parse.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any]]
            Return value description.

        """
        rows: List[Dict[str, Any]] = []
        max_pairs = 0

        with open(self.path, "r") as fh_count:
            total_lines = sum(1 for _ in fh_count)
        with open(self.path, "r") as fh:
            for line_i, raw in enumerate(fh, start=1):
                if self._reporter and (line_i % 200 == 0 or line_i == total_lines):
                    self._reporter("load", line_i, total_lines, "Parsing eregime.in")
                s = raw.strip()
                if not s or s.startswith("#"):
                    continue

                parts = s.split()
                if len(parts) < 4:
                    raise ValueError(
                        f"Malformed line (need at least iter, field_zones, direction, field): {raw!r}"
                    )

                # iter and number of zones
                it = int(float(parts[0]))
                zones = int(float(parts[1]))

                # parse direction/field pairs
                tail = parts[2:]
                if len(tail) % 2 != 0:
                    raise ValueError(f"Direction/field tokens must be pairs: {raw!r}")

                n_pairs = len(tail) // 2
                max_pairs = max(max_pairs, zones, n_pairs)

                rec: Dict[str, Any] = {"iter": it, "field_zones": zones}

                for i in range(n_pairs):
                    d = tail[2 * i]
                    e = float(tail[2 * i + 1])

                    if max(zones, n_pairs) <= 1:
                        rec["field_dir"] = d
                        rec["field"] = e
                    else:
                        rec[f"field_dir{i+1}"] = d
                        rec[f"field{i+1}"] = e

                rows.append(rec)

        if not rows:
            raise ValueError("No data lines found in eregime file.")

        # Build final column set
        if max_pairs <= 1:
            columns = ["iter", "field_zones", "field_dir", "field"]
        else:
            columns = ["iter", "field_zones"]
            for i in range(1, max_pairs + 1):
                columns += [f"field_dir{i}", f"field{i}"]

        # Normalize each record so all columns exist
        normed = [{col: r.get(col) for col in columns} for r in rows]
        df = pd.DataFrame(normed, columns=columns)

        meta = {
            "columns": list(df.columns),
            "max_field_zones": int(max_pairs),
            "n_records": int(len(df)),
        }
        if self._reporter:
            self._reporter("load", total_lines, total_lines, "Finished parsing eregime.in")
        return df, meta
