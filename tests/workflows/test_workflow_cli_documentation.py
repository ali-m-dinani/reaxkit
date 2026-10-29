from __future__ import annotations

import argparse
import ast
import importlib
import re
import shlex
from pathlib import Path

import pytest

import reaxkit.workflows
from reaxkit.core.runtime.cli_policy import add_execution_arguments


WORKFLOW_ROOT = Path(reaxkit.workflows.__file__).parent
TIMESERIES_MODULES = sorted(path.stem for path in (WORKFLOW_ROOT / "timeseries").glob("get_*.py"))


@pytest.mark.parametrize("module_name", TIMESERIES_MODULES)
def test_timeseries_help_documents_arguments_and_parseable_examples(module_name):
    module = importlib.import_module(f"reaxkit.workflows.timeseries.{module_name}")
    parser = argparse.ArgumentParser()
    module.build_parser(parser, command=module.COMMAND)
    add_execution_arguments(parser)

    assert issubclass(parser.formatter_class, argparse.RawDescriptionHelpFormatter)
    assert "\n\n" in parser.description
    assert "Examples:\n  1. " in parser.description
    assert parser.format_help()
    for action in parser._actions:
        if action.dest == "help" or action.help == argparse.SUPPRESS:
            continue
        assert action.help, action.dest
        assert action.help.count("Example:") == 1, action.dest
        assert re.search(r"Example: .+, .+", action.help), action.dest

    examples = [line.strip() for line in parser.description.splitlines() if line.strip().startswith("reaxkit ")]
    assert examples
    for example in examples:
        tokens = shlex.split(example)
        assert tokens[1].replace("-", "_") == module.COMMAND
        args = parser.parse_args(tokens[2:])
        assert module.build_request(args) is not None


def test_literal_workflow_argument_help_includes_examples():
    failures = []
    for path in sorted(WORKFLOW_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "add_argument":
                continue
            help_value = next((keyword.value for keyword in node.keywords if keyword.arg == "help"), None)
            if help_value is None:
                failures.append(f"{path.relative_to(WORKFLOW_ROOT)}:{node.lineno}: missing help")
            elif isinstance(help_value, ast.Constant) and isinstance(help_value.value, str):
                if "Example:" not in help_value.value:
                    failures.append(f"{path.relative_to(WORKFLOW_ROOT)}:{node.lineno}: missing example")
    assert not failures, "\n".join(failures)


def test_timeseries_dispatcher_examples_cover_all_analysis_tasks():
    from reaxkit.workflows.timeseries import timeseries_workflow

    parser = argparse.ArgumentParser()
    timeseries_workflow.build_parser(parser)
    tasks = set()
    for line in parser.description.splitlines():
        if not line.strip().startswith("reaxkit "):
            continue
        args = parser.parse_args(shlex.split(line.strip())[2:])
        task, request = timeseries_workflow._resolve_task_and_request(args)
        tasks.add(task)
        if task == "trajectory_displacement_series":
            assert request.atom_ids == tuple(range(1, 21))
    assert tasks == {
        "simulation_series", "trajectory_coordinate_series", "trajectory_displacement_series",
        "charge_series", "cell_dimensions", "electric_field_series", "eregime_series",
        "partial_energy_series", "restraint_series", "molecular_frequency_series",
        "molecular_totals_series", "geometry_optimization_data",
    }


@pytest.mark.parametrize("command", [
    "get_ffield_diagnostics_sensitivity", "get_ffield_diagnostics_evolution",
    "parameter_optimization_tornado", "merge-ffield",
])
def test_force_field_help_examples_use_available_choices(command):
    from reaxkit.workflows.file_tools import ffield_workflow

    parser = argparse.ArgumentParser()
    ffield_workflow.build_parser(parser, command=command)
    assert parser.format_help()
    for action in parser._actions:
        if action.dest not in {"plot", "report_format"}:
            continue
        snippet = action.help.split("Example:", 1)[1].split(", which", 1)[0]
        tokens = shlex.split(snippet)
        assert tokens[0] in action.option_strings
        assert tokens[1] in action.choices
