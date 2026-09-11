"""
Tests for eregime generator (eregime.in).

These tests validate the explicit-row writer:
- writes the expected header
- writes rows in the correct column order (start, V_index, direction, magnitude)
- normalizes/validates direction
- returns the output Path
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import pytest

from reaxkit.engine.reaxff.generators.eregime_generator import (
    _write_a_given_eregime as write_a_given_eregime,
    gen_eregime,
)
from reaxkit.workflows.file_tools.eregime_workflow import build_parser


def _read_lines(p: Path) -> list[str]:
    return p.read_text(encoding="utf-8").splitlines()


def _read_data(p: Path) -> list[tuple[int, int, str, float]]:
    return [
        (int(parts[0]), int(parts[1]), parts[2], float(parts[3]))
        for line in _read_lines(p)[2:]
        if (parts := line.split())
    ]


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


def test_sinusoid_ten_points_returns_to_baseline_three_times(tmp_path: Path):
    out = tmp_path / "eregime.in"

    gen_eregime(
        out,
        profile_type="sin",
        max_magnitude=0.35,
        points_per_cycle=10,
        iteration_step=500,
        num_cycles=1,
        direction="z",
    )

    rows = _read_data(out)
    assert len(rows) == 10
    assert [row[0] for row in rows] == list(range(0, 5000, 500))
    assert [index for index, row in enumerate(rows) if row[3] == 0.0] == [0, 4, 9]
    assert all(row[3] > 0.0 for row in rows[1:4])
    assert all(row[3] < 0.0 for row in rows[5:9])
    assert max(row[3] for row in rows) == pytest.approx(0.35)
    assert min(row[3] for row in rows) == pytest.approx(-0.35)


def test_sinusoid_cycles_share_boundary_without_duplicate_values(tmp_path: Path):
    out = tmp_path / "eregime.in"

    gen_eregime(
        out,
        profile_type="sin",
        max_magnitude=0.35,
        points_per_cycle=10,
        iteration_step=500,
        num_cycles=2,
        dc_offset=0.1,
    )

    rows = _read_data(out)
    assert len(rows) == 2 * (10 - 1) + 1
    assert [index for index, row in enumerate(rows) if row[3] == 0.1] == [0, 4, 9, 13, 18]
    assert all(first[3] != second[3] for first, second in zip(rows, rows[1:]))
    assert rows[-1][0] == 9000


def test_sinusoid_seventeen_points_uses_equal_quarter_cycle_increments(tmp_path: Path):
    out = tmp_path / "eregime.in"

    gen_eregime(
        out,
        profile_type="sin",
        max_magnitude=0.35,
        points_per_cycle=17,
        iteration_step=500,
        num_cycles=2,
    )

    rows = _read_data(out)
    magnitudes = [row[3] for row in rows]
    assert len(rows) == 2 * (17 - 1) + 1
    assert [index for index, value in enumerate(magnitudes) if value == 0.0] == [0, 8, 16, 24, 32]
    assert all(first != second for first, second in zip(magnitudes, magnitudes[1:]))
    for quarter_start in (0, 4, 8, 12):
        changes = [
            magnitudes[index + 1] - magnitudes[index]
            for index in range(quarter_start, quarter_start + 4)
        ]
        assert changes == pytest.approx([changes[0]] * 4)


def test_sinusoid_odd_point_count_samples_both_peaks(tmp_path: Path):
    out = tmp_path / "eregime.in"

    gen_eregime(
        out,
        profile_type="sin",
        max_magnitude=0.35,
        points_per_cycle=9,
        iteration_step=500,
        num_cycles=1,
    )

    magnitudes = [row[3] for row in _read_data(out)]
    assert max(magnitudes) == pytest.approx(0.35)
    assert min(magnitudes) == pytest.approx(-0.35)
    assert [index for index, value in enumerate(magnitudes) if value == 0.0] == [0, 4, 8]


def test_sinusoid_step_angle_remains_available(tmp_path: Path):
    out = tmp_path / "eregime.in"

    gen_eregime(
        out,
        profile_type="sin",
        max_magnitude=0.35,
        step_angle=0.4,
        iteration_step=500,
        num_cycles=1,
    )

    assert len(_read_data(out)) == round(2.0 * math.pi / 0.4) + 1


def test_sinusoid_sampling_flags_are_mutually_exclusive():
    parser = build_parser(argparse.ArgumentParser(), command="gen_eregime")

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--type",
                "sin",
                "--iteration-step",
                "500",
                "--points-per-cycle",
                "10",
                "--step-angle",
                "0.4",
            ]
        )


def test_sinusoid_help_documents_odd_counts_and_seventeen_point_example():
    parser = build_parser(argparse.ArgumentParser(), command="gen_eregime")
    help_text = parser.format_help()

    assert (
        "For equal-duration positive and negative halves, use an odd count such as 9 or 11. "
        "With 10 points there are nine time intervals, so one half necessarily has one additional interval."
        in help_text
    )
    assert (
        "reaxkit gen_eregime --type sin --output eregime.in --max-magnitude 0.35 "
        "--points-per-cycle 17 --iteration-step 500 --num-cycles 2 --direction z --V 1 --copy-to-dot"
        in help_text
    )
