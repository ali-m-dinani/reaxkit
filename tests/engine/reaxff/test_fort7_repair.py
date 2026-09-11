"""Regression tests for repairs of overflowed fort.7 integer fields."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from reaxkit.engine.reaxff.generators.fort7_repair import (
    _GeometryFrame,
    _cell_matrix,
    _fix_data_line,
    _format_data_line,
    repair_fort7,
)
from reaxkit.workflows.file_tools.fort7_workflow import build_parser


def test_fix_data_line_writes_fused_atom_type_and_first_neighbor_split() -> None:
    # Atom type 1 and neighbor id 20594 were fused into ``120594``.  The nine
    # zero neighbor slots complete a frame whose header declares ten bonds.
    line = " ".join(
        ["20596", "120594", *(["0"] * 9), "1", "0.829", *(["0.000"] * 9), "1.665", "1.068", "-1.153"]
    ) + "\n"

    repaired, status = _fix_data_line(line, n_bonds=10)

    assert status == "fixed"
    fields = repaired.split()
    assert fields[:3] == ["20596", "1", "20594"]
    assert fields[3:12] == ["0"] * 9
    assert fields[12] == "1"
    # The parser requires atom number, type, ten neighbors, and molecule id
    # to all be separate integers.
    assert [int(value) for value in fields[:13]] == [20596, 1, 20594, *([0] * 9), 1]
    assert repaired.startswith("20596    1 20594")


def test_fix_data_line_leaves_already_separated_row_unchanged() -> None:
    line = "    1    2    3    8  882 7364    0    1  0.537  0.546  0.546  0.545  0.000  4.013  0.000  1.188\n"

    repaired, status = _fix_data_line(line, n_bonds=5)

    assert status == "unchanged"
    assert repaired == line


def test_fix_data_line_uses_periodic_geometry_to_resolve_ambiguous_ids() -> None:
    # ``1234`` may be split as (1, 234) or (12, 34).  Atoms 12 and 34 are
    # nearest to atom 3 only after atom 12 is wrapped across the x boundary.
    coordinates = [(5.0, 5.0, 5.0)] * 234
    coordinates[2] = (0.1, 0.0, 0.0)
    coordinates[11] = (9.9, 0.0, 0.0)
    coordinates[33] = (0.3, 0.0, 0.0)
    frame = _GeometryFrame(
        iteration=0,
        coordinates=tuple(coordinates),
        cell=_cell_matrix((10.0, 10.0, 10.0), (90.0, 90.0, 90.0)),
    )
    line = "3 2 1234 1 0.500 0.500 1.000 -0.200\n"

    repaired, status = _fix_data_line(line, n_bonds=2, frame=frame)

    assert status == "fixed"
    assert repaired.split()[:5] == ["3", "2", "12", "34", "1"]
    assert repaired == "    3    2   12   34    1  0.500  0.500  1.000 -0.200\n"


def test_known_atom_3_ambiguity_uses_geometry_not_digit_pattern() -> None:
    coordinates = [(50.0, 50.0, 50.0)] * 19202
    coordinates[2] = (0.0, 0.0, 0.0)
    for neighbor, coordinate in {
        1: (1.0, 0.0, 0.0),
        6: (0.0, 1.0, 0.0),
        1206: (0.0, 0.0, 1.0),
        19202: (1.0, 1.0, 0.0),
    }.items():
        coordinates[neighbor - 1] = coordinate
    frame = _GeometryFrame(iteration=0, coordinates=tuple(coordinates), cell=None)
    line = "   3    2    1    6 120619202    0    1  0.537  0.546  0.546  0.546  0.000  2.183  1.000 -1.254\n"

    repaired, status = _fix_data_line(line, n_bonds=5, frame=frame)

    assert status == "fixed"
    assert repaired.split()[:8] == ["3", "2", "1", "6", "1206", "19202", "0", "1"]


def test_ambiguous_ids_are_not_guessed_without_matching_geometry() -> None:
    line = "   3    2    1    6 120619202    0    1  0.537  0.546  0.546  0.546  0.000  2.183  1.000 -1.254\n"

    repaired, status = _fix_data_line(line, n_bonds=5, frame=None)

    assert status == "unresolved"
    assert repaired == line


def test_repairs_incorrect_output_from_legacy_repairer() -> None:
    coordinates = [(50.0, 50.0, 50.0)] * 19202
    coordinates[2] = (0.0, 0.0, 0.0)
    for neighbor, coordinate in {
        1: (1.0, 0.0, 0.0),
        6: (0.0, 1.0, 0.0),
        1206: (0.0, 0.0, 1.0),
        19202: (1.0, 1.0, 0.0),
    }.items():
        coordinates[neighbor - 1] = coordinate
    frame = _GeometryFrame(iteration=0, coordinates=tuple(coordinates), cell=None)
    legacy_output = "3 2 1 61 2061 9202 0 1 0.537 0.546 0.546 0.546 0.000 2.183 1.000 -1.254\n"

    repaired, status = _fix_data_line(legacy_output, n_bonds=5, frame=frame)

    assert status == "fixed"
    assert repaired.split()[:8] == ["3", "2", "1", "6", "1206", "19202", "0", "1"]


def test_repaired_row_uses_fort7_column_alignment() -> None:
    repaired = _format_data_line(
        ["1", "3", "3", "81", "2081", "9204", "0", "1"],
        ["0.537", "0.546", "0.546", "0.546", "0.000", "4.023", "0.000", "1.227"],
    )

    assert repaired == (
        "    1    3    3   81 2081 9204    0    1"
        "  0.537  0.546  0.546  0.546  0.000  4.023  0.000  1.227\n"
    )


def test_cli_defaults_to_required_xmolout_file_in_current_directory() -> None:
    parser = build_parser(argparse.ArgumentParser(), command="repair_fort7")

    args = parser.parse_args([])

    assert args.xmolout == "xmolout"
    assert "the file must exist" in parser.format_help()


def test_repair_requires_xmolout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "fort.7").write_text("", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="required xmolout"):
        repair_fort7()


def test_repair_displays_dynamic_progress_bar(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fort7 = tmp_path / "fort.7"
    xmolout = tmp_path / "xmolout"
    output = tmp_path / "fixed"
    fort7.write_text(
        "2 sim Iteration: 0 #Bonds: 1\n"
        "    1    1    2    1  1.000  1.000  0.000 -0.100\n"
        "    2    1    1    1  1.000  1.000  0.000 -0.100\n",
        encoding="utf-8",
    )
    xmolout.write_text(
        "2\nsim 0 -1.0 10 10 10 90 90 90\nH 0 0 0\nH 1 0 0\n",
        encoding="utf-8",
    )

    repair_fort7(fort7, output, xmolout_file=xmolout, progress_every=1)

    progress_output = capsys.readouterr().err
    assert "Repairing fort.7" in progress_output
    assert "100%" in progress_output


def test_repair_rejects_xmolout_without_matching_iteration(tmp_path: Path) -> None:
    fort7 = tmp_path / "fort.7"
    xmolout = tmp_path / "xmolout"
    fort7.write_text("2 sim Iteration: 0 #Bonds: 1\n", encoding="utf-8")
    xmolout.write_text(
        "2\nsim 1 -1.0 10 10 10 90 90 90\nH 0 0 0\nH 1 0 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no frame matching fort.7 iteration 0"):
        repair_fort7(fort7, tmp_path / "fixed", xmolout_file=xmolout)
