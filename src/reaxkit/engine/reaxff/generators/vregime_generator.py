"""
Volume-regime (vregime.in) file generators.

This module provides utilities for generating ReaxFF ``vregime.in`` files,
which define how simulation cell dimensions and angles are modified over
time during molecular dynamics simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


__all__ = [
    "VRegimeSampleSpec",
    "VREGIME_GENERATOR_REGISTRY",
    "generate_sample_vregime",
    "write_vregime",
    "write_sample_vregime",
]


@dataclass(frozen=True)
class VRegimeSampleSpec:
    n_rows: int = 5


def _build_sample_rows() -> list[dict[str, Any]]:
    return [
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
    ]


def generate_sample_vregime(spec: VRegimeSampleSpec = VRegimeSampleSpec()) -> str:
    """
    Generate sample ``vregime.in`` text with fixed-width, left-aligned columns.
    """
    width_start = 6
    width_v = 4
    width_type = 6
    width_change = 12
    width_rescale = 8
    sep = " "

    def _pad(value: str, width: int) -> str:
        return str(value)[:width].ljust(width)

    def _fmt_start(value: Any) -> str:
        return f"{int(value):04d}"

    def _fmt_change(value: Any, decimals: int = 6) -> str:
        return f"{float(value):.{decimals}f}"

    header1 = "#Volume regimes"
    header2 = (
        _pad("#start", width_start)
        + sep
        + _pad("#V", width_v)
        + sep
        + _pad("type1", width_type)
        + sep
        + _pad("change/it", width_change)
        + sep
        + _pad("rescale", width_rescale)
        + sep
        + _pad("type 2", width_type)
        + sep
        + _pad("change/it", width_change)
        + sep
        + _pad("rescale", width_rescale)
    )

    def format_row(row: dict[str, Any]) -> str:
        terms = list(row.get("terms", []))
        vcount = len(terms)
        line = _pad(_fmt_start(row.get("start", 0)), width_start) + sep + _pad(str(vcount), width_v)
        for term in terms:
            line += (
                sep
                + _pad(term.get("type", ""), width_type)
                + sep
                + _pad(_fmt_change(term.get("change", 0.0)), width_change)
                + sep
                + _pad(term.get("rescale", "y"), width_rescale)
            )
        return line.rstrip()

    rows = _build_sample_rows()[: spec.n_rows]
    lines = [header1, header2.rstrip()]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines) + "\n"


def write_vregime(
    out_path: str | Path = "vregime.in",
    spec: VRegimeSampleSpec = VRegimeSampleSpec(),
) -> Path:
    """
    Write generated ``vregime.in`` text to disk.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(generate_sample_vregime(spec), encoding="utf-8", newline="\n")
    return out_path


def write_sample_vregime(
    out_path: str | Path = "vregime.in",
    *,
    n_rows: int = 5,
) -> Path:
    """
    Backward-compatible wrapper for writing sample ``vregime.in`` text.
    """
    return write_vregime(out_path=out_path, spec=VRegimeSampleSpec(n_rows=n_rows))


VREGIME_GENERATOR_REGISTRY: dict[str, dict[str, Any]] = {
    "vregime_sample": {
        "label": "Volume Regime Sample",
        "default_filename": "vregime.in",
        "spec_type": VRegimeSampleSpec,
        "generate": generate_sample_vregime,
        "write": write_vregime,
    }
}
