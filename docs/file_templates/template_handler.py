"""
Template ReaxFF file handler.

This module provides a minimal, extensible base for implementing parsers for
ReaxFF input/output files. A handler is responsible for reading a file and
exposing its contents as structured tabular data; analysis and plotting should
live in analyzer/workflow modules.

Typical use cases include:

- parsing a file into a summary ``pandas.DataFrame`` (one row per iteration)
- optionally collecting per-frame tables in ``self._frames``
- applying lightweight cleaning steps (e.g., de-duplicating iterations)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

from reaxkit.io.base_handler import FileHandler


class TemplateHandler(FileHandler):
    """
    Parser template for ReaxFF ``<filetype>`` files.

    This class illustrates the standard ReaxKit handler pattern: read a raw file,
    parse it into a summary table, optionally store per-frame tables, and expose
    simple accessors for downstream analyzers and workflows.

    Parsed Data
    -----------
    Summary table
        One row per record (often per iteration), returned by ``dataframe()``,
        with columns:
        [``iteration``, ``energy``, ``a``, ``b``, ``c``]

    Per-frame data
        Stored in ``self._frames`` (optional), where each frame is a
        ``pandas.DataFrame`` with file-specific columns.

    Notes
    -----
    - Replace the placeholder parsing logic in ``_parse()`` with file-specific
      rules for your target ReaxFF file.
    """

    def __init__(self, file_path: str | Path = "<filetype>"):
        """
        Create a handler instance for a ReaxFF file.

        Parameters
        ----------
        file_path : str or pathlib.Path, optional
            Path to the target file (default: ``"<filetype>"``).
        """
        super().__init__(file_path)
        self._frames: List[pd.DataFrame] = []  # file-specific (optional)
        self._n_records: Optional[int] = None

    def _parse(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        # Replace with actual parsing logic for <filetype>.
        sim_rows: List[list] = []
        frames: List[pd.DataFrame] = []

        sim_cols = ["iteration", "energy", "a", "b", "c"]

        with open(self.path, "r", encoding="utf-8") as fh:
            for _line in fh:
                # Parse header/metadata and per-record rows here.
                # Append parsed records to sim_rows and (optionally) frames.
                pass

        df = pd.DataFrame(sim_rows, columns=sim_cols)

        # Example clean step: drop duplicate iterations (keep the last).
        if "iteration" in df.columns and not df.empty:
            keep_idx = df.drop_duplicates("iteration", keep="last").index
            frames = [frames[i] for i in keep_idx] if frames else []
            df = df.loc[keep_idx].reset_index(drop=True)

        self._frames = frames
        meta: Dict[str, Any] = {"n_records": len(df), "n_frames": len(frames)}
        return df, meta

    def n_frames(self) -> int:
        """
        Return the number of available frames.

        Returns
        -------
        int
            Number of per-frame tables stored in ``self._frames``.
        """
        return int(self.metadata().get("n_frames", 0))

    def frame(self, i: int) -> Dict[str, Any]:
        """
        Return a per-frame record for index ``i``.

        Parameters
        ----------
        i : int
            Zero-based frame index.

        Returns
        -------
        dict[str, Any]
            Minimal per-frame record with keys: ``index``, ``iteration``, ``energy``.

        Examples
        --------
        >>> h = TemplateHandler("<filetype>")
        >>> rec = h.frame(0)
        >>> rec["iteration"]
        """
        df = self.dataframe()
        row = df.iloc[i]
        return {"index": i, "iteration": row.get("iteration"), "energy": row.get("energy")}

    def iter_frames(self, step: int = 1) -> Iterator[Dict[str, Any]]:
        """
        Yield per-frame records with an optional stride.

        Parameters
        ----------
        step : int, optional
            Stride between yielded frames (default: 1).

        Yields
        ------
        dict[str, Any]
            Per-frame records as returned by :meth:`frame`.
        """
        for i in range(0, self.n_frames(), max(1, int(step))):
            yield self.frame(i)
