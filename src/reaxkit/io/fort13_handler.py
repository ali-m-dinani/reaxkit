"""handler for parsing and cleaning data in fort.13 file"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional
import pandas as pd

from reaxkit.io.file_handler import FileHandler


class Fort13Handler(FileHandler):
    """
    Handler for ReaxFF fort.13 files.

    **Purpose**: Reads total error values from ReaxFF force-field optimization output.

    **Outputs**:
      - `DataFrame`: columns → ['epoch', 'total_ff_error']
      - `metadata`: dictionary with n_records, min_error, max_error, mean_error
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
        """Return number of total-error entries."""
        return int(self.metadata().get("n_records", 0))

    def iter_errors(self, step: int = 1) -> Iterator[Dict[str, Any]]:
        """Iterate over error values with step size."""
        df = self.dataframe()
        for i in range(0, len(df), step):
            yield {
                "epoch": int(df.iloc[i]["epoch"]),
                "total_ff_error": float(df.iloc[i]["total_ff_error"]),
            }
