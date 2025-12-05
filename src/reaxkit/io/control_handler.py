"""handler for parsing and cleaning data in control file"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from reaxkit.io.file_handler import FileHandler


class ControlHandler(FileHandler):
    """
    Parser for ReaxFF control file, using the generic FileHandler interface.

    Output
    ------
    dataframe() : pd.DataFrame
        Columns: section | key | value | inline_comment

        - section:        "general", "md", "mm", "ff", "outdated"
        - key:            parameter name (lowercase)
        - value:          numeric (int/float) when possible, otherwise string
        - inline_comment: text after '#' on the original line (if any)

    metadata() : dict
        Flattened counts of parameters per section, e.g.:

        {
            "n_general":  5,
            "n_md":       12,
            "n_mm":       0,
            "n_ff":       3,
            "n_outdated": 1,
        }

    Attributes (for backward compatibility)
    ---------------------------------------
    general_parameters : dict
    md_parameters      : dict
    mm_parameters      : dict
    ff_parameters      : dict
    outdated_parameters: dict
    """

    def __init__(self, file_path: str | Path = "control"):
        super().__init__(file_path)

        # Backward-compatible per-section dicts
        self.general_parameters: Dict[str, Any] = {}
        self.md_parameters: Dict[str, Any] = {}
        self.mm_parameters: Dict[str, Any] = {}
        self.ff_parameters: Dict[str, Any] = {}
        self.outdated_parameters: Dict[str, Any] = {}

        # Keep old behavior: parse eagerly so dicts are populated
        self.parse()

    # ------------------------------------------------------------------
    # Internal parsing logic
    # ------------------------------------------------------------------
    def _parse(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Parse control file into a DataFrame and metadata.

        DataFrame columns:
            - section
            - key
            - value
            - inline_comment
        """
        section: str | None = None
        rows: list[dict[str, Any]] = []
        counts: Dict[str, int] = {
            "general": 0,
            "md": 0,
            "mm": 0,
            "ff": 0,
            "outdated": 0,
        }

        # Helper to map a header line to a normalized section name
        def header_to_section(line: str) -> str | None:
            # remove leading '#' and surrounding whitespace, then lower
            hdr = re.sub(r"^\s*#\s*", "", line).strip().lower()
            if hdr.startswith("general"):
                return "general"
            if hdr.startswith("md"):
                return "md"
            if hdr.startswith("mm"):
                return "mm"
            if hdr.startswith("ff"):
                return "ff"
            if hdr.startswith("outdated"):
                return "outdated"
            return None

        try:
            with open(self.path, "r") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue  # skip blank lines

                    # Section headers start with '#'
                    if line.startswith("#"):
                        maybe = header_to_section(line)
                        if maybe:
                            section = maybe
                        # otherwise it's just an unrecognized comment/header
                        continue

                    if not section:
                        # ignore content before any recognized section header
                        continue

                    # Match lines like: "<number> <key>"
                    # Examples: "0.25 timestep", "300.0 temperature", "5000 nsteps"
                    m = re.match(r"^([\d\-.Ee+]+)\s+([A-Za-z_][\w]*)", line)
                    if not m:
                        # not a parameter line; skip
                        continue

                    value_str, key = m.groups()
                    key = key.lower()

                    # Extract inline comment: text after '#' in the ORIGINAL raw line
                    hash_idx = raw.find("#")
                    if hash_idx != -1:
                        inline_comment = raw[hash_idx + 1 :].strip()
                    else:
                        inline_comment = ""

                    # Convert numeric if possible
                    try:
                        value: Any = float(value_str)
                        if isinstance(value, float) and value.is_integer():
                            value = int(value)
                    except ValueError:
                        value = value_str  # keep raw if not numeric

                    # Update backward-compatible dicts
                    if section == "general":
                        self.general_parameters[key] = value
                    elif section == "md":
                        self.md_parameters[key] = value
                    elif section == "mm":
                        self.mm_parameters[key] = value
                    elif section == "ff":
                        self.ff_parameters[key] = value
                    elif section == "outdated":
                        self.outdated_parameters[key] = value

                    # Record row for DataFrame
                    rows.append(
                        {
                            "section": section,
                            "key": key,
                            "value": value,
                            "inline_comment": inline_comment,
                        }
                    )

                    # Count per section
                    if section in counts:
                        counts[section] += 1
                    else:
                        counts[section] = 1

        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Control file not found at {self.path}")

        # Build DataFrame (even if empty)
        df = pd.DataFrame(
            rows,
            columns=["section", "key", "value", "inline_comment"],
        )

        # Flattened metadata: only number of parameters per section
        meta: dict[str, Any] = {
            "n_general": counts.get("general", 0),
            "n_md": counts.get("md", 0),
            "n_mm": counts.get("mm", 0),
            "n_ff": counts.get("ff", 0),
            "n_outdated": counts.get("outdated", 0),
        }

        return df, meta
