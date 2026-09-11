"""Tests for extracting individual structures from optimization GEO files."""

from __future__ import annotations

import argparse

import pytest

from reaxkit.cli.main import _canonicalize_direct_command
from reaxkit.core.registry.analysis_cli_routing_registry import (
    get_registered_analysis_commands,
)
from reaxkit.workflows.force_field_opt import get_force_field_opt_geo_files as workflow

MULTI_GEO = """BIOGRF 200
DESCRP molecule_one
REMARK non-periodic example
FORMAT ATOM
HETATM     1 C                   1.00000   2.00000   3.00000    C   1 1  0.00000
HETATM     2 O                   4.00000   5.00000   6.00000    O   1 1  0.00000
END

XTLGRF 200
DESCRP  bulk_e3_mp_2604
CRYSTX    7.17250    7.17250    7.17250   90.00000   90.00000   90.00000
FORMAT ATOM
HETATM     1 Mg                  0.00000   0.00000   0.00000 Mg     0 0  0.00000
HETATM     2 Te                  2.77737   2.77737   2.77737 Te     0 0  0.00000
END
"""


def test_command_is_registered_with_hyphen_alias() -> None:
    commands = get_registered_analysis_commands()
    assert "get_force_field_opt_geo_files" in commands
    assert _canonicalize_direct_command(
        ["reaxkit", "get-force-field-opt-geo-files"]
    )[1] == "get_force_field_opt_geo_files"


def test_extracts_each_geometry_as_geo_and_xyz(tmp_path) -> None:
    source = tmp_path / "geo"
    source.write_text(MULTI_GEO, encoding="utf-8")

    results = workflow.extract_force_field_opt_geometries(source, tmp_path / "extracted")

    assert [result.identifier for result in results] == ["molecule_one", "bulk_e3_mp_2604"]
    geometry = tmp_path / "extracted" / "bulk_e3_mp_2604"
    geo_text = (geometry / "bulk_e3_mp_2604.geo").read_text(encoding="utf-8")
    xyz_lines = (geometry / "bulk_e3_mp_2604.xyz").read_text(encoding="utf-8").splitlines()
    assert geo_text.startswith("XTLGRF 200\nDESCRP  bulk_e3_mp_2604")
    assert geo_text.rstrip().endswith("END")
    assert xyz_lines[:2] == ["2", "bulk_e3_mp_2604"]
    assert xyz_lines[2].split() == ["Mg", "0.0000000000", "0.0000000000", "0.0000000000"]
    assert xyz_lines[3].split() == ["Te", "2.7773700000", "2.7773700000", "2.7773700000"]


def test_identifier_filter_and_cli_runner(tmp_path, capsys) -> None:
    source = tmp_path / "geo"
    source.write_text(MULTI_GEO, encoding="utf-8")
    output = tmp_path / "selected"
    parser = workflow.build_parser(
        argparse.ArgumentParser(), command="get_force_field_opt_geo_files"
    )
    args = parser.parse_args(
        ["--geo", str(source), "--output", str(output), "--identifier", "bulk_e3_mp_2604"]
    )

    assert workflow.run_main("get_force_field_opt_geo_files", args) == 0
    assert not (output / "molecule_one").exists()
    assert (output / "bulk_e3_mp_2604" / "bulk_e3_mp_2604.xyz").is_file()
    assert "Extracted 1 geometries" in capsys.readouterr().out


def test_refuses_to_overwrite_existing_outputs(tmp_path) -> None:
    source = tmp_path / "geo"
    source.write_text(MULTI_GEO, encoding="utf-8")
    output = tmp_path / "extracted"
    workflow.extract_force_field_opt_geometries(source, output)

    with pytest.raises(FileExistsError, match="--overwrite"):
        workflow.extract_force_field_opt_geometries(source, output)


def test_preserves_duplicate_identifiers_with_numbered_folders(tmp_path) -> None:
    source = tmp_path / "geo"
    duplicate = MULTI_GEO + MULTI_GEO.split("\n\n", maxsplit=1)[0] + "\n"
    source.write_text(duplicate, encoding="utf-8")

    results = workflow.extract_force_field_opt_geometries(source, tmp_path / "extracted")

    molecule_results = [result for result in results if result.identifier == "molecule_one"]
    assert [result.occurrence for result in molecule_results] == [1, 2]
    assert molecule_results[0].directory.name == "molecule_one"
    assert molecule_results[1].directory.name == "molecule_one__2"
    assert (molecule_results[1].directory / "molecule_one__2.geo").is_file()


def test_reports_missing_identifier(tmp_path) -> None:
    source = tmp_path / "geo"
    source.write_text(MULTI_GEO, encoding="utf-8")

    with pytest.raises(ValueError, match="not found"):
        workflow.extract_force_field_opt_geometries(
            source,
            tmp_path / "extracted",
            identifiers=["does_not_exist"],
        )
