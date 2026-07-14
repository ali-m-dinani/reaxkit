"""Tests for ReaxFF isomer representative detection."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from reaxkit.analysis.molecular_analysis.reaxff_isomer_representatives_detection import (
    detect_reaxff_isomer_representatives,
    parse_reaxff_isomer_representative_control,
)
from reaxkit.workflows.file_tools import isomer_workflow


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "reaxff_isomer_representatives_detection"


def _fixture_paths() -> tuple[Path, Path, Path]:
    return (
        FIXTURE_DIR / "fort.7",
        FIXTURE_DIR / "xmolout",
        FIXTURE_DIR / "control_params",
    )


def test_parse_reaxff_isomer_representative_control_reads_legacy_control_params() -> None:
    _, _, control_path = _fixture_paths()

    control = parse_reaxff_isomer_representative_control(control_path)

    assert control.atom_map == {"C": 1, "H": 2, "O": 3, "B": 4}
    assert control.input_formula == {"C": 8, "H": 13, "O": 3, "B": 5}
    assert control.atom_type_counts == {1: 8, 2: 13, 3: 3, 4: 5}
    assert control.total_atoms == 29
    assert control.isomer_run == 1
    assert control.isomer_prefixname == "C8H13O3B5"


def test_detect_reaxff_isomer_representatives_writes_expected_xmolout_from_trimmed_fixture(tmp_path: Path) -> None:
    fort7, xmolout, control = _fixture_paths()
    output_dir = tmp_path / "out"

    result = detect_reaxff_isomer_representatives(
        fort7_path=fort7,
        xmolout_path=xmolout,
        control_path=control,
        output_dir=output_dir,
    )

    assert result.output_xmolout_isomers == output_dir / "xmolout_isomers"
    assert result.isomer_dir is None
    assert result.log_path == output_dir / "isomer_run_log.txt"
    assert list(result.table["structure_name"]) == [
        "C8H13O3B5_0",
        "C8H13O3B5_1",
        "C8H13O3B5_2",
        "C8H13O3B5_3",
        "C8H13O3B5_4",
        "C8H13O3B5_5",
    ]
    assert list(result.table["iteration"]) == [0, 200, 1500, 2800, 3800, 15100]
    assert list(result.table["molecule_no"]) == [1, 1, 1, 1, 1, 1]
    assert list(result.table["atom_count"]) == [29, 29, 29, 29, 29, 29]
    assert result.output_xmolout_isomers.read_text(encoding="utf-8") == (
        FIXTURE_DIR / "expected_xmolout_isomers"
    ).read_text(encoding="utf-8")


def test_detect_reaxff_isomer_representatives_honors_max_representatives(tmp_path: Path) -> None:
    fort7, xmolout, control = _fixture_paths()
    output_dir = tmp_path / "limited"

    result = detect_reaxff_isomer_representatives(
        fort7_path=fort7,
        xmolout_path=xmolout,
        control_path=control,
        output_dir=output_dir,
        max_representatives=2,
    )

    assert list(result.table["structure_name"]) == ["C8H13O3B5_0", "C8H13O3B5_1"]
    assert list(result.table["iteration"]) == [0, 200]
    expected_first_two = "\n".join(
        (FIXTURE_DIR / "expected_xmolout_isomers").read_text(encoding="utf-8").splitlines()[:62]
    ) + "\n"
    assert output_dir.joinpath("xmolout_isomers").read_text(encoding="utf-8") == expected_first_two


def test_detect_reaxff_isomer_representatives_validates_required_input_files(tmp_path: Path) -> None:
    fort7, xmolout, control = _fixture_paths()

    with pytest.raises(FileNotFoundError, match="fort.7 file not found"):
        detect_reaxff_isomer_representatives(
            fort7_path=tmp_path / "missing_fort.7",
            xmolout_path=xmolout,
            control_path=control,
            output_dir=tmp_path / "out",
        )

    with pytest.raises(FileNotFoundError, match="xmolout file not found"):
        detect_reaxff_isomer_representatives(
            fort7_path=fort7,
            xmolout_path=tmp_path / "missing_xmolout",
            control_path=control,
            output_dir=tmp_path / "out",
        )


def test_detect_isomer_representatives_workflow_runs_file_generation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fort7, xmolout, control = _fixture_paths()
    output_dir = tmp_path / "workflow_out"
    args = argparse.Namespace(
        fort7=str(fort7),
        xmolout=str(xmolout),
        control=str(control),
        output_dir=str(output_dir),
        write_isomer_dirs=True,
        no_isomer_dirs=False,
        max_representatives=3,
    )

    assert isomer_workflow.run_main("detect-isomer-representatives", args) == 0

    captured = capsys.readouterr()
    assert "detect-isomer-representatives: detected 3 isomer representatives" in captured.out
    assert (output_dir / "isomers" / "C8H13O3B5_2" / "xmolout").is_file()
    assert not (output_dir / "isomers" / "C8H13O3B5_3" / "xmolout").exists()
