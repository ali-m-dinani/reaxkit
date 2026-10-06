from __future__ import annotations

import argparse
import importlib
import sys

import pytest

from reaxkit.engine.reaxff.generators.trajectory_frame import extract_frame
from reaxkit.workflows import trajectory_workflow


FIRST = "2\nslab 0 -100 10 11 12 80 90 100\nAl 1.23456789 2 3 0.4\nN 4 5 6 -0.4\n"
SECOND = "2\nslab 23465 -90 103.30 59.64 78.26 90 90 90\nAl 7 8 9\nN 10 11 12\n"


@pytest.mark.parametrize("frame", ["last", -1, 1])
def test_extract_selected_cell_and_coordinates(tmp_path, frame):
    source = tmp_path / "xmolout"
    source.write_text(FIRST + SECOND)
    xyz, geo = extract_frame(source, frame=frame, xyz_file=tmp_path / "out.xyz", geo_file=tmp_path / "out.geo")
    assert xyz.read_text() == SECOND
    crystx = next(line for line in geo.read_text().splitlines() if line.startswith("CRYSTX"))
    assert list(map(float, crystx.split()[1:])) == [103.30, 59.64, 78.26, 90, 90, 90]
    atoms = [line.split() for line in geo.read_text().splitlines() if line.startswith("HETATM")]
    assert [atom[2] for atom in atoms] == ["Al", "N"]
    assert [list(map(float, atom[3:6])) for atom in atoms] == [[7, 8, 9], [10, 11, 12]]


@pytest.mark.parametrize("tail", ["2\n", "2\nslab 4", "2\nslab 4 -80 10 11 12 90 90 90\nAl 0 0 0\n", "2\nslab 4 -80 10 11 12 90 90 90\nAl 0 0 0\nN 1 2 3e-"])
def test_last_ignores_incomplete_tail(tmp_path, tail):
    source = tmp_path / "xmolout"
    source.write_text(FIRST + SECOND + tail)
    xyz, _ = extract_frame(source, xyz_file=tmp_path / "out.xyz", geo_file=tmp_path / "out.geo")
    assert xyz.read_text() == SECOND
    with pytest.raises(ValueError, match="No complete frame"):
        extract_frame(source, frame=2, xyz_file=tmp_path / "missing.xyz", geo_file=tmp_path / "missing.geo")
    assert not (tmp_path / "missing.xyz").exists()


def test_early_frame_preserves_precision_and_ignores_later_damage(tmp_path):
    source = tmp_path / "xmolout"
    source.write_text(FIRST + "invalid later data\n")
    xyz, geo = extract_frame(source, frame=0, xyz_file=tmp_path / "first.xyz", geo_file=tmp_path / "first.geo")
    assert "Al 1.23456789 2 3\nN 4 5 6\n" in xyz.read_text()
    crystx = next(line for line in geo.read_text().splitlines() if line.startswith("CRYSTX"))
    assert list(map(float, crystx.split()[1:])) == [10, 11, 12, 80, 90, 100]


@pytest.mark.parametrize("contents", ["", "2\n", FIRST.replace("10 11 12", "nan 11 12"), FIRST.replace("Al 1.23456789", "Al nan"), FIRST.replace("80 90 100", "10 10 170")])
def test_invalid_input_writes_no_outputs(tmp_path, contents):
    source = tmp_path / "xmolout"
    source.write_text(contents)
    with pytest.raises(ValueError):
        extract_frame(source, xyz_file=tmp_path / "out.xyz", geo_file=tmp_path / "out.geo")
    assert not (tmp_path / "out.xyz").exists()
    assert not (tmp_path / "out.geo").exists()


@pytest.mark.parametrize("frame", [-2, "bad", "1.5", 10])
def test_invalid_selector(tmp_path, frame):
    source = tmp_path / "xmolout"
    source.write_text(FIRST)
    with pytest.raises(ValueError):
        extract_frame(source, frame=frame, xyz_file=tmp_path / "out.xyz", geo_file=tmp_path / "out.geo")


def test_source_collision(tmp_path):
    source = tmp_path / "xmolout"
    source.write_text(FIRST)
    with pytest.raises(ValueError, match="different files"):
        extract_frame(source, xyz_file=source, geo_file=tmp_path / "out.geo")
    assert source.read_text() == FIRST


def test_workflow_run_directory_and_alias(tmp_path):
    (tmp_path / "xmolout").write_text(FIRST + SECOND)
    parser = trajectory_workflow.build_parser(argparse.ArgumentParser(), command="extract-frame")
    args = parser.parse_args(["--run-dir", str(tmp_path), "--output", str(tmp_path / "restart")])
    assert trajectory_workflow.run_main("extract-frame", args) == 0
    assert (tmp_path / "restart.xyz").read_text() == SECOND
    assert (tmp_path / "restart.geo").exists()


def test_cli_extract_frame(tmp_path, monkeypatch):
    cli_main = importlib.import_module("reaxkit.cli.main")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "xmolout").write_text(FIRST + SECOND)
    monkeypatch.setattr(sys, "argv", ["reaxkit", "extract-frame", "--file", "xmolout", "--frame", "0", "--output", "restart"])
    assert cli_main.main() == 0
    assert (tmp_path / "restart.xyz").read_text().startswith("2\nslab 0 ")
    assert (tmp_path / "restart.geo").exists()
