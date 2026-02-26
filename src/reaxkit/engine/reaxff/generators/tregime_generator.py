"""
Temperature-regime (tregime.in) file generators.

This module provides utilities for generating ReaxFF ``tregime.in`` files,
which define temperature control zones and thermostat parameters used
during molecular dynamics simulations.

Typical use cases include:

- creating a valid sample ``tregime.in`` file for new simulations
- inspecting column layout and formatting rules
- using a template as a starting point for custom temperature schedules
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Sequence


def write_sample_tregime(
    out_path: str | Path = "tregime.in",
    *,
    n_rows: int = 3,
) -> None:
    """
    Write a sample tregime.in file with fixed-width, left-aligned columns.

    Works on
    --------
    ReaxFF temperature-regime input — ``tregime.in``

    Parameters
    ----------
    out_path : str or pathlib.Path, optional
        Output file path for the generated tregime file.
    n_rows : int, optional
        Number of example data rows to write.

    Returns
    -------
    None
        Writes a formatted ``tregime.in`` file to disk.

    Examples
    --------
    >>> from reaxkit.io.generators.tregime_generator import write_sample_tregime
    >>> write_sample_tregime("tregime.in", n_rows=2)
    """
    out_path = Path(out_path)

    # (name, width, kind)
    COLS = [
        ("#Start",   10, "int"),
        ("#Zones",   10, "int"),
        ("At1",      8,  "int"),
        ("At2",      8,  "int"),
        ("Tset1",    10, "float1"),
        ("Tdamp1",   10, "float1"),
        ("dT1/dt",   10, "float3"),
        ("At3",      8,  "int"),
        ("At4",      8,  "int"),
        ("Tset2",    10, "float1"),
        ("Tdamp2",   10, "float1"),
        ("dT2/dt",   10, "float3"),
    ]

    SEP = "  "  # two spaces between columns (keeps it readable)

    def _fmt_value(v: Any, kind: str) -> str:
        if kind == "int":
            return str(int(v))
        if kind == "float1":
            return f"{float(v):.1f}"
        if kind == "float3":
            return f"{float(v):.3f}"
        return str(v)

    def _pad_left(s: str, width: int) -> str:
        # left-align content within fixed width
        if len(s) > width:
            return s[:width]
        return s.ljust(width)

    def format_header() -> str:
        return SEP.join(_pad_left(name, width) for name, width, _ in COLS)

    def format_row(values: Dict[str, Any]) -> str:
        parts = []
        for name, width, kind in COLS:
            raw = _fmt_value(values.get(name, 0), kind)
            parts.append(_pad_left(raw, width))
        return SEP.join(parts)

    # Example rows
    rows: Sequence[Dict[str, Any]] = [
        {
            "#Start": 0, "#Zones": 2,
            "At1": 1, "At2": 50, "Tset1": 300.0, "Tdamp1": 100.0, "dT1/dt": 0.050,
            "At3": 51, "At4": 100, "Tset2": 600.0, "Tdamp2": 100.0, "dT2/dt": 0.100,
        },
        {
            "#Start": 5000, "#Zones": 1,
            "At1": 1, "At2": 100, "Tset1": 900.0, "Tdamp1": 200.0, "dT1/dt": 0.020,
            "At3": 0, "At4": 0, "Tset2": 0.0, "Tdamp2": 0.0, "dT2/dt": 0.000,
        },
    ][:n_rows]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="\n") as fh:
        fh.write(format_header() + "\n")
        for r in rows:
            fh.write(format_row(r) + "\n")

