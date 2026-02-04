"""
ReaxFF parameter search definition (params) handler.

This module provides a handler for parsing ReaxFF ``params`` files,
which define parameter indices, search intervals, bounds, and optional
inline comments used during force-field optimization.

Typical use cases include:

- inspecting parameter search spaces
- linking optimization parameters to force-field sections
- building interpretable training and sensitivity analyses
"""


from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from reaxkit.io.base_handler import FileHandler


class ParamsHandler(FileHandler):
    """
    Parser for ReaxFF parameter search definition files (``params``).

    This class parses ``params`` files and exposes parameter search
    definitions as a structured tabular dataset suitable for training,
    optimization, and diagnostics workflows.

    Parsed Data
    -----------
    Summary table
        One row per parameter entry, returned by ``dataframe()``, with columns:
        ["ff_section", "ff_section_line", "ff_parameter",
         "search_interval", "min_value", "max_value", "inline_comment"]

        The columns map to ReaxFF force-field definitions as follows:
        - ``ff_section``: force-field section identifier
          (1–7 → general, atom, bond, off-diagonal, angle, torsion, h-bond)
        - ``ff_section_line``: line index within the corresponding section
        - ``ff_parameter``: parameter index within that line

    Metadata
        Returned by ``metadata()``, containing:
        ["n_records", "n_frames"]

    Notes
    -----
    - Inline comments following ``!`` are preserved verbatim.
    - Lines with incorrect token counts raise a parsing error.
    - This handler is not frame-based; ``n_frames()`` always returns 0.
    """

    COLUMNS = [
        "ff_section",
        "ff_section_line",
        "ff_parameter",
        "search_interval",
        "min_value",
        "max_value",
        "inline_comment",
    ]

    def __init__(self, file_path: str | Path = "params.in"):
        super().__init__(file_path)

    def _parse(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Implementation of TemplateHandler._parse for params files.

        Returns
        -------
        df : DataFrame
            With columns: ff_section, ff_section_line, ff_parameter,
            search_interval, min_value, max_value, inline_comment.
        meta : dict
            Metadata with keys: n_records, n_frames.
        """
        rows: List[Dict[str, Any]] = []

        with open(self.path, "r") as fh:
            for raw_line in fh:
                line = raw_line.strip()

                # Skip empty lines and full-line comments
                if not line or line.startswith(("!", "#")):
                    continue

                # Split off inline comment at first "!"
                before, sep, comment = line.partition("!")
                inline_comment = comment.strip() if sep else ""

                # Numeric / token part
                tokens = before.split()
                if not tokens:
                    continue

                # Expect exactly 6 numeric tokens:
                # ff_section ff_section_line ff_parameter search_interval min_value max_value
                if len(tokens) != 6:
                    raise ValueError(
                        f"Expected 6 tokens in params line, got {len(tokens)}: {raw_line!r}"
                    )

                ff_section = int(tokens[0])
                ff_section_line = int(tokens[1])
                ff_parameter = int(tokens[2])
                search_interval = float(tokens[3])
                min_value = float(tokens[4])
                max_value = float(tokens[5])

                rows.append(
                    {
                        "ff_section": ff_section,
                        "ff_section_line": ff_section_line,
                        "ff_parameter": ff_parameter,
                        "search_interval": search_interval,
                        "min_value": min_value,
                        "max_value": max_value,
                        "inline_comment": inline_comment,
                    }
                )

        df = pd.DataFrame(rows, columns=self.COLUMNS)

        # No per-frame data for this file type
        self._frames = []

        meta: Dict[str, Any] = {
            "n_records": len(df),
            "n_frames": 0,
        }
        return df, meta
