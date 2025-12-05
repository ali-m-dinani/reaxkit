"""handler for parsing and cleaning data in params file"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from reaxkit.io.file_handler import FileHandler


class ParamsHandler(FileHandler):
    """
    Handler for ReaxFF params-like files, with lines such as:

        3 49  1  1.0000   45.0   180.0        !Zn-Pt bond parameters
        3 49  4  0.0100   -1.00   1.000
        4 28  1  0.0020   0.05    0.3         !Zn-Pt off-diagonal

    Parsed columns:
        ff_section        (int)
        ff_section_line   (int)
        ff_parameter      (int)
        search_interval   (float)
        min_value         (float)
        max_value         (float)
        inline_comment    (str, may be empty)
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
