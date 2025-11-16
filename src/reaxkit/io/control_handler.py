"""handler for parsing and cleaning data in control file"""
import re
from pathlib import Path
from typing import Dict


class ControlHandler:
    """
    Parser for ReaxFF control file.

    Sections supported (case/spacing-insensitive):
    - # General
    - # MD
    - # MM
    - # FF
    - # Outdated

    Attributes
    ----------
    general_parameters : dict
    md_parameters : dict
    mm_parameters : dict
    ff_parameters : dict
    outdated_parameters : dict
    """

    def __init__(self, file_path: str = "control"):
        self.file_path = Path(file_path)
        self.general_parameters: Dict[str, float] = {}
        self.md_parameters: Dict[str, float] = {}
        self.mm_parameters: Dict[str, float] = {}
        self.ff_parameters: Dict[str, float] = {}
        self.outdated_parameters: Dict[str, float] = {}
        self._parse()

    def _parse(self) -> None:
        """Parse control file into sections and store parameters."""
        section = None

        # Helper to map a header line to a normalized section name
        def header_to_section(line: str):
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
            with open(self.file_path, "r") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue  # skip blank lines

                    if line.startswith("#"):
                        # Detect and switch section; do NOT break on Outdated
                        maybe = header_to_section(line)
                        if maybe:
                            section = maybe
                        # otherwise it's just a comment/header we don't recognize
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
                    try:
                        value = float(value_str)
                        if value.is_integer():
                            value = int(value)
                    except ValueError:
                        value = value_str  # keep raw if not numeric

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

        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Control file not found at {self.file_path}")
