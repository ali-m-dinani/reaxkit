"""
Volume-regime (vregime.in) file generators.

This module provides utilities for generating ReaxFF ``vregime.in`` files,
which define how simulation cell dimensions and angles are modified over
time during molecular dynamics simulations.

Typical use cases include:

- creating a valid sample ``vregime.in`` file for new simulations
- inspecting column layout and formatting rules for volume schedules
- using a template as a starting point for custom volume/strain protocols
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence


def write_sample_vregime(
    out_path: str | Path = "vregime.in",
    *,
    n_rows: int = 5,
) -> None:
    """
    Write a sample vregime.in file with fixed-width, left-aligned columns.

    Works on
    --------
    ReaxFF volume-regime input — ``vregime.in``

    Parameters
    ----------
    out_path : str or pathlib.Path, optional
        Output file path for the generated vregime file.
    n_rows : int, optional
        Number of example data rows to write.

    Returns
    -------
    None
        Writes a formatted ``vregime.in`` file to disk.

    Examples
    --------
    >>> from reaxkit.io.generators.vregime_generator import write_sample_vregime
    >>> write_sample_vregime("vregime.in", n_rows=3)
    """
    out_path = Path(out_path)

    # Fixed widths (LEFT aligned)
    W_START = 6     # gives room for 4-digit start + spaces
    W_V = 4         # '#V' column
    W_TYPE = 6      # 'alfa', 'beta', 'a', 'b', etc.
    W_CHANGE = 12   # change/it numeric field
    W_RESCALE = 8   # 'y' / 'n'

    SEP = " "  # single-space like your working example

    def _pad(s: str, w: int) -> str:
        s = "" if s is None else str(s)
        return s[:w].ljust(w)

    def _fmt_start(v: Any) -> str:
        # match the example: 0000, 0100, 0200, ...
        return f"{int(v):04d}"

    def _fmt_change(v: Any, *, decimals: int = 6) -> str:
        # match the example's precision style (0.050000, -0.010000, etc.)
        return f"{float(v):.{decimals}f}"

    # Header
    header1 = "#Volume regimes"
    header2 = (
        _pad("#start", W_START)
        + SEP
        + _pad("#V", W_V)
        + SEP
        + _pad("type1", W_TYPE)
        + SEP
        + _pad("change/it", W_CHANGE)
        + SEP
        + _pad("rescale", W_RESCALE)
        + SEP
        + _pad("type 2", W_TYPE)
        + SEP
        + _pad("change/it", W_CHANGE)
        + SEP
        + _pad("rescale", W_RESCALE)
    )

    # Sample rows (mirrors your screenshot)
    # Each row: {"start": int, "terms": [{"type": str, "change": float, "rescale": "y|n"}, ...]}
    rows: List[Dict[str, Any]] = [
        {
            "start": 0,
            "terms": [
                {"type": "alfa", "change": 0.050000, "rescale": "y"},
                {"type": "beta", "change": -0.050000, "rescale": "y"},
            ],
        },
        {
            "start": 100,
            "terms": [
                {"type": "beta", "change": 0.050000, "rescale": "y"},
                {"type": "alfa", "change": -0.050000, "rescale": "y"},
            ],
        },
        {
            "start": 200,
            "terms": [
                {"type": "a", "change": 0.010000, "rescale": "y"},
                {"type": "b", "change": -0.010000, "rescale": "y"},
            ],
        },
        {
            "start": 300,
            "terms": [
                {"type": "a", "change": -0.010000, "rescale": "y"},
                {"type": "b", "change": 0.010000, "rescale": "y"},
            ],
        },
        {
            "start": 400,
            "terms": [
                {"type": "a", "change": -0.010000, "rescale": "y"},
                {"type": "alfa", "change": 0.050000, "rescale": "y"},
                {"type": "b", "change": 0.010000, "rescale": "y"},
                {"type": "beta", "change": 0.050000, "rescale": "y"},
            ],
        },
    ][:n_rows]

    def format_row(r: Dict[str, Any]) -> str:
        terms = list(r.get("terms", []))
        vcount = len(terms)

        line = (
            _pad(_fmt_start(r.get("start", 0)), W_START)
            + SEP
            + _pad(str(vcount), W_V)
        )

        # Append each (type, change, rescale) group
        for t in terms:
            line += (
                SEP
                + _pad(str(t.get("type", "")), W_TYPE)
                + SEP
                + _pad(_fmt_change(t.get("change", 0.0)), W_CHANGE)
                + SEP
                + _pad(str(t.get("rescale", "y")), W_RESCALE)
            )

        return line.rstrip()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="\n") as fh:
        fh.write(header1 + "\n")
        fh.write(header2.rstrip() + "\n")
        for r in rows:
            fh.write(format_row(r) + "\n")

