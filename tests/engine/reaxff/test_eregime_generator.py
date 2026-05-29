"""
Tests for eregime generator (eregime.in).

These tests validate the explicit-row writer:
- writes the expected header
- writes rows in the correct column order (start, V_index, direction, magnitude)
- normalizes/validates direction
- returns the output Path
"""

from __future__ import annotations

from pathlib import Path
import re
import pytest

from reaxkit.io.generators.eregime_generator import write_a_given_eregime


def _read_lines(p: Path) -> list[str]:
    return p.read_text(encoding="utf-8").splitlines()


def test_write_a_given_eregime_writes_header_and_rows(tmp_path: Path):
    out = tmp_path / "eregime.in"
    rows = [
        (0, 1, "z", 0.01),
        (100, 1, "z", -0.01),
    ]

    result = write_a_given_eregime(out, rows)
    assert isinstance(result, Path)
    assert result.exists()

    lines = _read_lines(out)
    assert lines[0].startswith("#Electric field regimes")
    assert lines[1].startswith("#start")
    assert len(lines) == 2 + len(rows)

    # Check first data row structure: iter, V, dir, magnitude
    first = lines[2]
    parts = first.split()
    assert parts[0] == "0"
    assert parts[1] == "1"
    assert parts[2].lower() == "z"
    assert pytest.approx(float(parts[3]), rel=1e-12, abs=1e-12) == 0.01


def test_direction_is_normalized_to_lowercase(tmp_path: Path):
    out = tmp_path / "eregime.in"
    write_a_given_eregime(out, [(0, 1, "Z", 0.05)])

    lines = _read_lines(out)
    parts = lines[2].split()
    assert parts[2] == "z"


def test_invalid_direction_raises(tmp_path: Path):
    out = tmp_path / "eregime.in"
    with pytest.raises(ValueError):
        write_a_given_eregime(out, [(0, 1, "q", 0.05)])


def test_format_contains_fixed_columns(tmp_path: Path):
    """
    The writer uses aligned columns; we don't lock exact spacing, but we verify:
    - iteration and V_index are integers
    - direction is one of x/y/z
    - magnitude is a float with 6 decimals (as formatted by the writer)
    """
    out = tmp_path / "eregime.in"
    write_a_given_eregime(out, [(12, 3, "x", 1.23456789)])

    line = _read_lines(out)[2]
    # Example format: "    12     3        x               1.234568"
    m = re.match(r"^\s*(\d+)\s+(\d+)\s+([xyz])\s+([-+]?\d+\.\d{6})\s*$", line)
    assert m is not None
    assert m.group(1) == "12"
    assert m.group(2) == "3"
    assert m.group(3) == "x"
    assert m.group(4) == "1.234568"
