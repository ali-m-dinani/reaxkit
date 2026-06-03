"""
ReaxFF simulation summary (summary.txt) handler.

This module provides a handler for parsing ReaxFF ``summary.txt`` files,
which report per-iteration thermodynamic and system-level quantities
during MD or minimization runs.

Typical use cases include:

- tracking energy, temperature, pressure, and density versus iteration
- extracting time-series data for plotting or analysis
- validating simulation stability and convergence

**Usage context**

- ReaxFF parsing: Read ReaxFF text outputs into normalized tabular structures.
- Workflow ingestion: Provide canonical handler interfaces used by adapters/workflows.
- Diagnostics/export: Preserve parsed metadata for reporting and downstream conversion.
"""


from __future__ import annotations
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from reaxkit.engine.reaxff.io.base import BaseHandler


class SummaryHandler(BaseHandler):
    """
    Parser for ReaxFF simulation summary files (``summary.txt``).

    This class parses ``summary.txt`` outputs and exposes per-iteration
    simulation summaries as a canonical, numeric time series.

    Parsed Data
    -----------
    Summary table
        One row per iteration, returned by ``dataframe()``, with columns
        determined by the detected column count:

        - 8 columns:
          ["iter", "nmol", "time", "E_pot", "V", "T", "P", "D"]

        - 9 columns:
          ["iter", "nmol", "time", "E_pot", "V", "T", "P", "D", "elap_time"]

    Metadata
        Returned by ``metadata()``, containing:
        ["n_records", "columns", "has_time", "source_file"]

    Notes
    -----
    - Banner and header lines starting with ``REAX`` or ``Iteration`` are ignored.
    - Rows are parsed as whitespace-delimited numeric data with no in-file header.
    - Duplicate iteration indices are resolved by keeping the last entry.
    - This handler represents a scalar-per-iteration time-series file.
    """

    def __init__(self, file_path: str | Path = "summary.txt", reporter=None) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        file_path : str | Path
            Parameter description.

        """
        super().__init__(file_path)
        self._reporter = reporter

    @staticmethod
    def _canonical_names(ncols: int) -> List[str]:
        """
         canonical names.

        Parameters
        ----------
        ncols : int
            Parameter description.

        Returns
        -------
        List[str]
            Return value description.

        """
        if ncols == 8:
            return ["iter", "nmol", "time", "E_pot", "V", "T", "P", "D"]
        if ncols == 9:
            return ["iter", "nmol", "time", "E_pot", "V", "T", "P", "D", "elap_time"]
        raise ValueError(
            f"Unsupported number of columns in summary data: {ncols}. "
            f"Expected 8 or 9."
        )

    def _parse(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
         parse.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any]]
            Return value description.

        """
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"Summary file not found: {p}")

        # Read, filter out banners/headers, keep only data lines.
        with p.open("r") as fh:
            raw_lines = fh.readlines()

        data_lines: List[str] = []
        total_lines = len(raw_lines)
        for line_i, ln in enumerate(raw_lines, start=1):
            s = ln.strip()
            if not s:
                if self._reporter:
                    self._reporter("load", line_i, total_lines, "Scanning summary.txt")
                continue
            s_lower = s.lower()
            # Skip banner/header lines only; keep every other line
            if s_lower.startswith("reax"):
                if self._reporter:
                    self._reporter("load", line_i, total_lines, "Scanning summary.txt")
                continue
            if s_lower.startswith("iteration"):
                if self._reporter:
                    self._reporter("load", line_i, total_lines, "Scanning summary.txt")
                continue
            if not s[0].isdigit():  # skipping comment or warning lines that may occur at the end of the file
                if self._reporter:
                    self._reporter("load", line_i, total_lines, "Scanning summary.txt")
                continue
            data_lines.append(ln)
            if self._reporter:
                self._reporter("load", line_i, total_lines, "Scanning summary.txt")

        if not data_lines:
            raise ValueError(
                "No data lines found after removing 'REAX…' and 'Iteration…' headers."
            )

        # Parse as whitespace-delimited, no header
        data_str = "".join(data_lines).strip()
        df = pd.read_csv(StringIO(data_str), sep=r"\s+", header=None, engine="python")

        # Assign canonical names based on column count
        names = self._canonical_names(df.shape[1])
        df.columns = names

        # Cleanup
        df = df.dropna(how="all").reset_index(drop=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "iter" in df.columns:
            df = df.dropna(subset=["iter"]).reset_index(drop=True)
            df = df.drop_duplicates("iter", keep="last").reset_index(drop=True)

        meta: Dict[str, Any] = {
            "n_records": int(len(df)),
            "columns": list(df.columns),
            "has_time": "time" in df.columns,
            "source_file": str(p),
        }
        return df, meta

    # Convenience accessors on canonical schema
    def fields(self) -> List[str]:
        """
        Fields.

        Returns
        -------
        List[str]
            Return value description.

        """
        return list(self.dataframe().columns)

    def has_times(self) -> bool:
        """
        Has times.

        Returns
        -------
        bool
            Return value description.

        """
        return "time" in self.dataframe().columns

    def iterations(self) -> pd.Series:
        """
        Iterations.

        Returns
        -------
        pd.Series
            Return value description.

        """
        df = self.dataframe()
        if "iter" not in df.columns:
            raise KeyError("'iter' column not found in summary.")
        return df["iter"]
