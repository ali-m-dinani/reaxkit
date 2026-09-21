import argparse

import numpy as np
import pandas as pd
import pytest

from reaxkit.cli.main import _canonicalize_direct_command
from reaxkit.core.registry.analysis_cli_routing_registry import (
    get_registered_analysis_commands,
)
from reaxkit.domain.data_models import TrajectoryData
from reaxkit.workflows.electrostatics import dielectric_constant_workflow


def _parser():
    return dielectric_constant_workflow.build_parser(
        argparse.ArgumentParser(), command="get-dielectric-constant"
    )


def test_cli_exposes_units_and_documents_every_flag():
    parser = _parser()
    args = parser.parse_args(
        [
            "--input", "dipole.xlsx",
            "--time-unit", "fs",
            "--dipole-unit", "debye",
            "--temperature", "300",
            "--volume", "1000",
            "--volume-unit", "angstrom3",
        ]
    )

    assert args.time_column == "t"
    assert args.dipole_column == "dipole"
    assert args.dipole_kind == "total"
    assert args.volume_method == "hull"
    assert parser.formatter_class is argparse.RawTextHelpFormatter
    assert "\n\nExamples:\n" in parser.description
    for action in parser._actions:
        if action.dest != "help":
            assert action.help and action.help.count("Example:") == 1


def test_workflow_reads_excel_and_writes_three_result_sheets(tmp_path):
    source = tmp_path / "dipole.xlsx"
    output = tmp_path / "result.xlsx"
    pd.DataFrame(
        {"t": np.arange(8, dtype=float), "dipole": [-2, -1, 0, 1, 2, 1, 0, -1]}
    ).to_excel(source, index=False)
    args = _parser().parse_args(
        [
            "--input", str(source),
            "--time-unit", "fs",
            "--dipole-unit", "debye",
            "--temperature", "300",
            "--volume", "1000",
            "--volume-unit", "angstrom3",
            "--output", str(output),
        ]
    )

    assert dielectric_constant_workflow.run_main("get-dielectric-constant", args) == 0
    assert output.is_file()
    assert set(pd.ExcelFile(output).sheet_names) == {
        "summary", "spectrum", "autocorrelation"
    }


def test_command_is_registered_with_aliases():
    spec = get_registered_analysis_commands()["get-dielectric-constant"]
    assert spec.aliases == ("get_dielectric_constant", "dielectric-constant")

    for alias in spec.aliases:
        argv = _canonicalize_direct_command(["reaxkit", alias])
        assert argv[1] == "get-dielectric-constant"


def test_workflow_calculates_default_hull_volume_from_trajectory(tmp_path, monkeypatch):
    source = tmp_path / "dipole.xlsx"
    output = tmp_path / "hull_result.xlsx"
    pd.DataFrame(
        {"t": np.arange(8, dtype=float), "dipole": [-2, -1, 0, 1, 2, 1, 0, -1]}
    ).to_excel(source, index=False)
    cube = np.asarray(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0], [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
        ]
    )
    trajectory = TrajectoryData(
        positions=cube[None, :, :],
        elements=["C"] * 8,
        atom_ids=list(range(1, 9)),
        iterations=np.asarray([0]),
    )
    monkeypatch.setattr(
        dielectric_constant_workflow,
        "load_trajectory_for_volume",
        lambda _args: trajectory,
    )
    args = _parser().parse_args(
        [
            "--input", str(source),
            "--trajectory", str(tmp_path / "xmolout"),
            "--time-unit", "fs",
            "--dipole-unit", "debye",
            "--temperature", "300",
            "--output", str(output),
        ]
    )

    assert dielectric_constant_workflow.run_main("get-dielectric-constant", args) == 0
    summary = pd.read_excel(output, sheet_name="summary")
    assert summary.loc[0, "volume method"] == "hull"
    assert summary.loc[0, "volume (angstrom3)"] == pytest.approx(1.0)
