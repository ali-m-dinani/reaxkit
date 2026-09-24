from __future__ import annotations

import argparse

import numpy as np

from reaxkit.analysis.ferroelectrics.charge_extxyz import ChargeExtendedXYZRequest
from reaxkit.core.registry.analysis_cli_routing_registry import get_registered_analysis_commands
from reaxkit.workflows.ferroelectrics import charge_extxyz_workflow as workflow


def test_charge_extxyz_parser_and_command_registration() -> None:
    parser = workflow.build_parser(argparse.ArgumentParser(), command=workflow.COMMAND)
    args = parser.parse_args(
        [
            "--frames", "2:8:2",
            "--every", "2",
            "--precision", "10",
            "--output", "view.xyz",
            "--include-electric-field",
            "--field-direction", "x",
        ]
    )

    assert workflow.build_request(args) == ChargeExtendedXYZRequest(
        frames=[2, 4, 6],
        every=2,
        precision=10,
        include_electric_field=True,
        field_direction="x",
    )
    route = get_registered_analysis_commands()[workflow.COMMAND]
    assert route.module_path == "reaxkit.workflows.ferroelectrics.charge_extxyz_workflow"
    assert "generate_charge_extxyz" in route.aliases


def test_default_output_is_in_write_trajectory_with_charges_folder(tmp_path) -> None:
    parser = workflow.build_parser(argparse.ArgumentParser(), command=workflow.COMMAND)
    args = parser.parse_args(
        ["--project-root", str(tmp_path), "--analysis-id", "run_1"]
    )

    assert workflow._output_path(args) == (
            tmp_path
            / "analysis"
            / "write_trajectory_with_charges"
            / "run_1"
            / "charges_delta_charges.extxyz"
    ).resolve()

    shared = tmp_path / "shared_results"
    args = parser.parse_args(["--output-dir", str(shared)])
    assert workflow._output_path(args) == (shared / "charges_delta_charges.extxyz").resolve()


def test_charge_extxyz_workflow_runs_with_quick_reaxff_stream(tmp_path) -> None:
    xmolout = tmp_path / "xmolout"
    xmolout.write_text(
        """2
sim 0 0 10 10 10 90 90 90
Al 0 0 0
N 1 0 0
2
sim 10 0 10 10 10 90 90 90
Al 0 0 0.1
N 1 0 0.1
""",
        encoding="utf-8",
    )
    fort7 = tmp_path / "fort.7"
    fort7.write_text(
        """2 sim Iteration: 0 #Bonds: 0
1 1 1 0.0 0.0 0.1
2 2 1 0.0 0.0 -0.1
0.0 0.0 0.0 0.0
2 sim Iteration: 10 #Bonds: 0
1 1 1 0.0 0.0 0.25
2 2 1 0.0 0.0 -0.2
0.0 0.0 0.0 0.0
""",
        encoding="utf-8",
    )
    output = tmp_path / "ovito.extxyz"
    parser = workflow.build_parser(argparse.ArgumentParser(), command=workflow.COMMAND)
    args = parser.parse_args(
        [
            "--engine",
            "reaxff",
            "--input",
            str(tmp_path),
            "--fort7",
            str(fort7),
            "--xmolout",
            str(xmolout),
            "--output",
            str(output),
            "--project-root",
            str(tmp_path / "project"),
        ]
    )

    assert workflow.run_main(workflow.COMMAND, args) == 0
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 8
    assert np.isclose(float(lines[6].split()[-1]), 0.15)
