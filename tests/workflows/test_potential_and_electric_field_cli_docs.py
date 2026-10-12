"""CLI documentation rules for local electrostatics workflows."""

import argparse

import pytest

from reaxkit.workflows.electrostatics.potential_and_electric_field import (
    potential_and_electric_field_workflow,
    trajectory_workflow,
)


@pytest.mark.parametrize(
    ("workflow", "command"),
    [
        (potential_and_electric_field_workflow, "get-potential-and-electric-field"),
        (trajectory_workflow, "write-trajectory-with-potential-and-electric-field"),
    ],
)
def test_cli_documentation_follows_workflow_rules(workflow, command):
    parser = workflow.build_parser(argparse.ArgumentParser(), command=command)

    assert parser.formatter_class is argparse.RawTextHelpFormatter
    assert parser.description.startswith(("Calculate", "Write"))
    assert "\n\nExamples:\n" in parser.description
    assert f"reaxkit {command}" in parser.description

    for action in parser._actions:
        if action.dest == "help":
            continue
        assert action.help, f"{action.option_strings} has no help text"
        assert action.help.count("Example:") == 1, action.option_strings


def test_potential_field_parser_accepts_kymograph_controls():
    parser = potential_and_electric_field_workflow.build_parser(
        argparse.ArgumentParser(), command="get-potential-and-electric-field"
    )
    args = parser.parse_args([
        "--bin-axes", "z", "--bins", "40", "--plot-kymograph",
        "--kymograph-values", "external-potential", "total-local-field",
        "--kymograph-time-axis", "iteration", "--plot-field-component", "z",
    ])
    assert args.bin_axes == "z"
    assert args.plot_kymograph
    assert args.kymograph_values == ["external-potential", "total-local-field"]
    assert args.kymograph_time_axis == "iteration"


@pytest.mark.parametrize("workflow", [potential_and_electric_field_workflow, trajectory_workflow])
def test_potential_field_workflows_accept_ams(workflow):
    from reaxkit.workflows.electrostatics.potential_and_electric_field.common import runtime_arguments

    parser = workflow.build_parser(argparse.ArgumentParser(), command=workflow.COMMAND)
    args = parser.parse_args([
        "--engine", "ams", "--input", "reaxout.kf", "--fort78", "fort.78",
        "--frames", "0:54000:50",
    ])
    runtime = runtime_arguments(args)
    assert runtime["engine"] == "ams"
    assert runtime["input"] == "reaxout.kf"
    assert runtime["fort78"] == "fort.78"
    assert runtime["frames"] == ["0:54000:50"]
    assert parser.parse_args([]).engine == "reaxff"
    assert parser.parse_args([]).export_per_atom_csv is False
    assert parser.parse_args(["--export-per-atom-csv"]).export_per_atom_csv is True
