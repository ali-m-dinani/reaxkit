"""
Temperature-regime (tregime.in) file generators.

This module provides utilities for generating ReaxFF ``tregime.in`` files,
which define temperature control zones and thermostat parameters used
during molecular dynamics simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


__all__ = [
    "TRegimeSampleSpec",
    "TREGIME_GENERATOR_REGISTRY",
    "generate_sample_tregime",
    "write_tregime",
    "write_sample_tregime",
]


@dataclass(frozen=True)
class TRegimeSampleSpec:
    n_rows: int = 3


def _build_sample_rows() -> list[dict[str, Any]]:
    return [
        {
            "#Start": 0,
            "#Zones": 2,
            "At1": 1,
            "At2": 50,
            "Tset1": 300.0,
            "Tdamp1": 100.0,
            "dT1/dt": 0.050,
            "At3": 51,
            "At4": 100,
            "Tset2": 600.0,
            "Tdamp2": 100.0,
            "dT2/dt": 0.100,
        },
        {
            "#Start": 5000,
            "#Zones": 1,
            "At1": 1,
            "At2": 100,
            "Tset1": 900.0,
            "Tdamp1": 200.0,
            "dT1/dt": 0.020,
            "At3": 0,
            "At4": 0,
            "Tset2": 0.0,
            "Tdamp2": 0.0,
            "dT2/dt": 0.000,
        },
    ]


def generate_sample_tregime(spec: TRegimeSampleSpec = TRegimeSampleSpec()) -> str:
    """
    Generate sample ``tregime.in`` text with fixed-width, left-aligned columns.
    """
    cols = [
        ("#Start", 10, "int"),
        ("#Zones", 10, "int"),
        ("At1", 8, "int"),
        ("At2", 8, "int"),
        ("Tset1", 10, "float1"),
        ("Tdamp1", 10, "float1"),
        ("dT1/dt", 10, "float3"),
        ("At3", 8, "int"),
        ("At4", 8, "int"),
        ("Tset2", 10, "float1"),
        ("Tdamp2", 10, "float1"),
        ("dT2/dt", 10, "float3"),
    ]
    sep = "  "

    def _fmt_value(value: Any, kind: str) -> str:
        if kind == "int":
            return str(int(value))
        if kind == "float1":
            return f"{float(value):.1f}"
        if kind == "float3":
            return f"{float(value):.3f}"
        return str(value)

    def _pad_left(value: str, width: int) -> str:
        if len(value) > width:
            return value[:width]
        return value.ljust(width)

    def format_header() -> str:
        return sep.join(_pad_left(name, width) for name, width, _ in cols)

    def format_row(values: dict[str, Any]) -> str:
        parts = []
        for name, width, kind in cols:
            raw = _fmt_value(values.get(name, 0), kind)
            parts.append(_pad_left(raw, width))
        return sep.join(parts)

    rows = _build_sample_rows()[: spec.n_rows]
    lines = [format_header()]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines) + "\n"


def write_tregime(
    out_path: str | Path = "tregime.in",
    spec: TRegimeSampleSpec = TRegimeSampleSpec(),
) -> Path:
    """
    Write generated ``tregime.in`` text to disk.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(generate_sample_tregime(spec), encoding="utf-8", newline="\n")
    return out_path


def write_sample_tregime(
    out_path: str | Path = "tregime.in",
    *,
    n_rows: int = 3,
) -> Path:
    """
    Backward-compatible wrapper for writing sample ``tregime.in`` text.
    """
    return write_tregime(out_path=out_path, spec=TRegimeSampleSpec(n_rows=n_rows))


TREGIME_GENERATOR_REGISTRY: dict[str, dict[str, Any]] = {
    "tregime_sample": {
        "label": "Temperature Regime Sample",
        "default_filename": "tregime.in",
        "spec_type": TRegimeSampleSpec,
        "generate": generate_sample_tregime,
        "write": write_tregime,
    }
}
