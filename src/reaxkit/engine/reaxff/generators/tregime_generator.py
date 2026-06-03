"""
Temperature-regime (tregime.in) file generators.

This module provides utilities for generating ReaxFF ``tregime.in`` files,
which define temperature control zones and thermostat parameters used
during molecular dynamics simulations.

**Usage context**

- Template generation: Produce canonical text payloads for ReaxFF artifacts.
- File writing: Persist generated outputs to disk with stable formatting.
- Workflow integration: Support higher-level ReaxKit workflow commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


__all__ = [
    "TRegimeSampleSpec",
    "TREGIME_GENERATOR_REGISTRY",
    "gen_template_tregime",
]


@dataclass(frozen=True)
class TRegimeSampleSpec:
    """Represent TRegimeSampleSpec.

    Public class used by ReaxFF generator components.

    Fields
    ------
    n_rows : int
        Dataclass field.
    """
    n_rows: int = 3


def _build_sample_rows() -> list[dict[str, Any]]:
    """Build sample rows."""
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


def _gen_template_tregime_text(spec: TRegimeSampleSpec = TRegimeSampleSpec()) -> str:
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
        """Fmt value."""
        if kind == "int":
            return str(int(value))
        if kind == "float1":
            return f"{float(value):.1f}"
        if kind == "float3":
            return f"{float(value):.3f}"
        return str(value)

    def _pad_left(value: str, width: int) -> str:
        """Pad left."""
        if len(value) > width:
            return value[:width]
        return value.ljust(width)

    def format_header() -> str:
        """Format header.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Return value.

        Examples
        --------
        ```python
        # Example
        format_header(...)
        ```
        """
        return sep.join(_pad_left(name, width) for name, width, _ in cols)

    def format_row(values: dict[str, Any]) -> str:
        """Format row.

        Parameters
        ----------
        values : dict[str, Any]
            Input parameter.

        Returns
        -------
        str
            Return value.

        Examples
        --------
        ```python
        # Example
        format_row(...)
        ```
        """
        parts = []
        for name, width, kind in cols:
            raw = _fmt_value(values.get(name, 0), kind)
            parts.append(_pad_left(raw, width))
        return sep.join(parts)

    rows = _build_sample_rows()[: spec.n_rows]
    lines = [format_header()]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines) + "\n"


def _write_tregime(
    out_path: str | Path = "tregime.in",
    spec: TRegimeSampleSpec = TRegimeSampleSpec(),
) -> Path:
    """
    Write generated ``tregime.in`` text to disk.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_gen_template_tregime_text(spec), encoding="utf-8", newline="\n")
    return out_path


def gen_template_tregime(
    out_path: str | Path = "tregime.in",
    *,
    n_rows: int = 3,
) -> Path:
    """Gen template tregime.

    Parameters
    ----------
    out_path : str | Path, optional
        Input parameter.
    n_rows : int, optional
        Keyword-only parameter.

    Returns
    -------
    Path
        Return value.

    Examples
    --------
    ```python
    # Example
    gen_template_tregime(...)
    ```
    """
    return _write_tregime(out_path=out_path, spec=TRegimeSampleSpec(n_rows=n_rows))


TREGIME_GENERATOR_REGISTRY: dict[str, dict[str, Any]] = {
    "tregime_sample": {
        "label": "Temperature Regime Sample",
        "default_filename": "tregime.in",
        "spec_type": TRegimeSampleSpec,
        "generate": _gen_template_tregime_text,
        "write": gen_template_tregime,
    }
}
