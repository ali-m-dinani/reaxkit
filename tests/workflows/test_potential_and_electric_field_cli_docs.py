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
