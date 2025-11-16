"""template handler for parsing and cleaning data in any ReaxFF file"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator
import pandas as pd

from reaxkit.io.file_handler import FileHandler


class TemplateHandler(FileHandler):
    """
    Handler for <filetype> files.
    - Reads raw file, parses into summary DataFrame + per-frame data.
    - Does only parsing/cleaning, no analysis or plotting.
    """

    def __init__(self, file_path: str | Path = "<filetype>"):
        super().__init__(file_path)
        self._frames: List[pd.DataFrame] = []   # file-specific (optional)
        self._n_records: Optional[int] = None

    def _parse(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        # --- Replace with actual parsing logic for <filetype> ---
        sim_rows: List[list] = []
        frames: List[pd.DataFrame] = []

        # Example: columns to store summary
        sim_cols = ["iteration", "energy", "a", "b", "c"]

        # Example: parse loop
        with open(self.path, "r") as fh:
            for line in fh:
                vals = line.strip().split()
                # parse header/metadata
                # parse per-frame data
                # append to sim_rows / frames

        df = pd.DataFrame(sim_rows, columns=sim_cols)

        # Example clean step: drop duplicate iterations
        if "iteration" in df.columns and not df.empty:
            keep_idx = df.drop_duplicates("iteration", keep="last").index
            frames = [frames[i] for i in keep_idx]
            df = df.loc[keep_idx].reset_index(drop=True)

        self._frames = frames
        meta: Dict[str, Any] = {
            "n_records": len(df),
            "n_frames": len(frames),
        }
        return df, meta

    # ---- File-specific accessors
    def n_frames(self) -> int:
        return int(self.metadata().get("n_frames", 0))

    def frame(self, i: int) -> Dict[str, Any]:
        """Return per-frame data structure (customize for your file)."""
        df = self.dataframe()
        row = df.iloc[i]
        return {
            "index": i,
            "iteration": row.get("iteration"),
            "energy": row.get("energy"),
            # Add other fields as needed
        }

    def iter_frames(self, step: int = 1) -> Iterator[Dict[str, Any]]:
        for i in range(0, self.n_frames(), step):
            yield self.frame(i)
