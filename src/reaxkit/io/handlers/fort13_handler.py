"""
ReaxFF force-field optimization error (fort.13) handler.

This module provides a handler for parsing ReaxFF ``fort.13`` files,
which store the total force-field error values produced during
ReaxFF parameter optimization runs.

Typical use cases include:

- tracking optimization convergence
- comparing force-field parameter sets
- plotting total error versus optimization epoch
"""


from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional
import pandas as pd

from reaxkit.io.base_handler import FileHandler


class Fort13Handler(FileHandler):
    """
    Parser for ReaxFF force-field optimization output files (``fort.13``).

    This class parses ``fort.13`` files and exposes total force-field
    error values as a simple, iteration-indexed time series.

    Parsed Data
    -----------
    Summary table
        One row per optimization epoch, returned by ``dataframe()``,
        with columns:
        ["epoch", "total_ff_error"]

    Metadata
        Returned by ``metadata()``, containing:
        ["n_records", "min_error", "max_error", "mean_error"]

    Notes
    -----
    - Epoch indices are inferred from line order in the file.
    - Non-numeric or empty lines are ignored.
    - This handler represents a single-scalar-per-iteration data source.
    """

    def __init__(self, file_path: str | Path = "fort.13"):
        super().__init__(file_path)
        self._n_records: Optional[int] = None

    def _parse(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Parse fort.13 file into a summary DataFrame."""
        sim_rows: List[list] = []
        with open(self.path, "r") as fh:
            for idx, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    error_value = float(line)
                except ValueError:
                    continue
                sim_rows.append([idx, error_value])

        df = pd.DataFrame(sim_rows, columns=["epoch", "total_ff_error"])

        meta: Dict[str, Any] = {
            "n_records": len(df),
            "min_error": df["total_ff_error"].min() if not df.empty else None,
            "max_error": df["total_ff_error"].max() if not df.empty else None,
            "mean_error": df["total_ff_error"].mean() if not df.empty else None,
        }

        self._n_records = meta["n_records"]
        return df, meta

    # ---- Accessors ----
    def n_records(self) -> int:
        """
        Return the number of optimization epochs recorded in the file.

        Works on
        --------
        Fort13Handler — ``fort.13``

        Returns
        -------
        int
            Number of parsed optimization epochs.
        """
        return int(self.metadata().get("n_records", 0))

    def iter_errors(self, step: int = 1) -> Iterator[Dict[str, Any]]:
        """Iterate over total force-field error values with optional subsampling.

        Works on
        --------
        Fort13Handler — ``fort.13``

        Parameters
        ----------
        step : int, optional
            Step size for subsampling epochs (default: 1).

        Yields
        ------
        dict
            Dictionary with keys: ``epoch`` and ``total_ff_error``.

        Examples
        --------
        >>> h = Fort13Handler("fort.13")
        >>> for row in h.iter_errors(step=10):
        ...     print(row["epoch"], row["total_ff_error"])
        """
        df = self.dataframe()
        for i in range(0, len(df), step):
            yield {
                "epoch": int(df.iloc[i]["epoch"]),
                "total_ff_error": float(df.iloc[i]["total_ff_error"]),
            }
