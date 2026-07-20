"""Tests for isomer trainset generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from reaxkit.analysis.molecular_analysis.isomer_trainset import (
    create_isomer_trainset,
    parse_isomer_hf_output,
)
from reaxkit.engine.reaxff.io.trainset_handler import TrainsetHandler
from reaxkit.workflows.file_tools import isomer_trainset_workflow


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "isomer_trainset"


def test_parse_isomer_hf_output_reads_final_energy_and_geometry() -> None:
    record = parse_isomer_hf_output(FIXTURE_DIR / "isomers" / "C8H13O3B5_54" / "hf.out")

    assert record.structure_name == "C8H13O3B5_54"
    assert record.energy_hartree == pytest.approx(-662.94648287463)
    assert record.energy_kcal == pytest.approx(-416004.884522, abs=1e-6)
    assert len(record.atoms) == 29
    assert record.atoms[0].element == "B"
    assert record.atoms[0].x == "13.3560340619"
    assert record.atoms[-1].element == "O"


def test_create_isomer_trainset_writes_legacy_outputs(tmp_path: Path) -> None:
    result = create_isomer_trainset(
        job_dir=FIXTURE_DIR / "isomers",
        output_dir=tmp_path / "trainset_outputs",
    )

    assert len(result.records) == 4
    assert len(result.skipped) == 1
    assert result.skipped[0].structure_name == "C8H13O3B5_incomplete"

    assert result.trainset_path.read_text(encoding="utf-8") == "\n".join(
        [
            "ENERGY",
            " 1.0  + C8H13O3B5_25/1    - C8H13O3B5_54/1    4.728014",
            " 1.0  + C8H13O3B5_1/1    - C8H13O3B5_54/1    55.324065",
            " 1.0  + C8H13O3B5_0/1    - C8H13O3B5_54/1    55.348802",
            "ENDENERGY",
            "",
        ]
    )
    energy_table = TrainsetHandler(result.trainset_path).energy_terms()
    assert len(energy_table) == 3
    assert energy_table["lit"].tolist() == pytest.approx([4.728014, 55.324065, 55.348802])

    geo_text = result.geo_path.read_text(encoding="utf-8")
    assert geo_text.startswith("29\nC8H13O3B5_0\nB 12.3279350928 11.0348747836 13.0890729005 \n")
    assert "\n29\nC8H13O3B5_54\nB 13.3560340619 13.9223624437 13.0892586517 \n" in geo_text

    assert result.composition_path.read_text(encoding="utf-8").splitlines() == [
        "C8H13O3B5_0 4.86308e-45",
        "C8H13O3B5_1 5.08998e-45",
        "C8H13O3B5_25 0.000163938",
        "C8H13O3B5_54 100",
    ]

    log_text = result.log_path.read_text(encoding="utf-8")
    assert "C8H13O3B5_54 -416004.884522" in log_text
    assert "C8H13O3B5_incomplete C8H13O3B5_incomplete does not have final geometry" in log_text
    assert "4 structures are present in trainset file." in log_text


def test_create_isomer_trainset_can_require_all_complete(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not all isomer jobs"):
        create_isomer_trainset(
            job_dir=FIXTURE_DIR / "isomers",
            output_dir=tmp_path / "trainset_outputs",
            require_all_complete=True,
        )


def test_create_isomer_trainset_refuses_nonempty_output_without_force(tmp_path: Path) -> None:
    output_dir = tmp_path / "trainset_outputs"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty"):
        create_isomer_trainset(
            job_dir=FIXTURE_DIR / "isomers",
            output_dir=output_dir,
        )


def test_isomer_trainset_workflow_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = argparse.Namespace(
        job_dir=str(FIXTURE_DIR / "isomers"),
        output_dir=str(tmp_path / "trainset_outputs"),
        hf_output_name="hf.out",
        geo_file="geo",
        trainset_file="trainset.in",
        composition_file="composition.txt",
        log_file="out_trainset_log.txt",
        weight=1.0,
        reference_composition=100.0,
        temperature=273.0,
        gas_constant=1.987,
        require_all_complete=False,
        force=False,
    )

    assert isomer_trainset_workflow.run_main("create-isomer-trainset", args) == 0

    captured = capsys.readouterr()
    assert "create-isomer-trainset: processed 5 isomers; included 4; skipped 1" in captured.out
    assert (tmp_path / "trainset_outputs" / "trainset.in").is_file()
