"""handler for parsing and cleaning data in summary.txt file"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
from io import StringIO

from reaxkit.io.file_handler import FileHandler


class SummaryHandler(FileHandler):
    """
    Parser for ReaxFF 'summary.txt' outputs.

    Simple behavior:
      • Ignore any line starting with 'REAX' or 'Iteration' (case-insensitive).
      • Treat the rest as whitespace-delimited numeric rows with NO header in-file.
      • Canonicalize column names by detected column count:
           8 -> [iter, nmol, time, E_pot, V, T, P, D]
           9 -> [iter, nmol, time, E_pot, V, T, P, D, elap_time]
      • Coerce numerics, drop empty rows, drop duplicate 'iter' (keep last).
    """

    def __init__(self, file_path: str | Path = "summary.txt") -> None:
        super().__init__(file_path)

    @staticmethod
    def _canonical_names(ncols: int) -> List[str]:
        if ncols == 8:
            return ["iter", "nmol", "time", "E_pot", "V", "T", "P", "D"]
        if ncols == 9:
            return ["iter", "nmol", "time", "E_pot", "V", "T", "P", "D", "elap_time"]
        raise ValueError(
            f"Unsupported number of columns in summary data: {ncols}. "
            f"Expected 8 or 9."
        )

    def _parse(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"Summary file not found: {p}")

        # Read, filter out banners/headers, keep only data lines.
        with p.open("r") as fh:
            raw_lines = fh.readlines()

        data_lines: List[str] = []
        for ln in raw_lines:
            s = ln.strip()
            if not s:
                continue
            s_lower = s.lower()
            # Skip banner/header lines only; keep every other line
            if s_lower.startswith("reax"):
                continue
            if s_lower.startswith("iteration"):
                continue
            if not s[0].isdigit():  # skipping comment or warning lines that may occur at the end of the file
                continue
            data_lines.append(ln)

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
        return list(self.dataframe().columns)

    def has_times(self) -> bool:
        return "time" in self.dataframe().columns

    def iterations(self) -> pd.Series:
        df = self.dataframe()
        if "iter" not in df.columns:
            raise KeyError("'iter' column not found in summary.")
        return df["iter"]
