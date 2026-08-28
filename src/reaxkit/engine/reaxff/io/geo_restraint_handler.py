"""Parse per-structure restraint declarations from a ReaxFF ``geo`` file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from reaxkit.engine.reaxff.io.base import BaseHandler


_COLUMNS = [
    "descriptor",
    "restraint_type",
    "coordinate",
    "atom1",
    "atom2",
    "atom3",
    "descriptor_line_number",
    "restraint_line_number",
]


class GeoRestraintHandler(BaseHandler):
    """Return one row per BOND or ANGLE restraint in a multi-structure geo file."""

    def __init__(self, file_path: str | Path = "geo", reporter=None):
        super().__init__(file_path)
        self._reporter = reporter

    def _parse(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        descriptor = ""
        descriptor_line_number: int | None = None
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, raw in enumerate(handle, start=1):
                stripped = raw.strip()
                upper = stripped.upper()
                if upper.startswith("DESCRP"):
                    descriptor = stripped[6:].strip()
                    descriptor_line_number = line_number
                    continue
                if not descriptor or not (
                    upper.startswith("BOND RESTRAINT")
                    or upper.startswith("ANGLE RESTRAINT")
                ):
                    continue
                parts = stripped.split()
                restraint_type = parts[0].lower()
                try:
                    atom1 = int(parts[2])
                    atom2 = int(parts[3])
                    if restraint_type == "bond":
                        atom3 = pd.NA
                        coordinate = float(parts[4])
                    else:
                        atom3 = int(parts[4])
                        coordinate = float(parts[5])
                except (IndexError, TypeError, ValueError):
                    continue
                rows.append(
                    {
                        "descriptor": descriptor,
                        "restraint_type": restraint_type,
                        "coordinate": coordinate,
                        "atom1": atom1,
                        "atom2": atom2,
                        "atom3": atom3,
                        "descriptor_line_number": descriptor_line_number,
                        "restraint_line_number": line_number,
                    }
                )

        table = pd.DataFrame(rows, columns=_COLUMNS)
        metadata = {
            "n_restraints": len(table),
            "n_structures": int(table["descriptor"].nunique()) if not table.empty else 0,
            "restraint_types": (
                sorted(table["restraint_type"].dropna().astype(str).unique().tolist())
                if not table.empty
                else []
            ),
        }
        if self._reporter:
            self._reporter("load", 1, 1, "Finished parsing geo restraints")
        return table, metadata


__all__ = ["GeoRestraintHandler"]
