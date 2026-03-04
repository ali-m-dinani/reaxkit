"""Task-oriented workflow for control-file analyses and utilities."""

from __future__ import annotations

import argparse
from typing import Callable

from reaxkit.analysis import control as _control_tasks  # noqa: F401
from reaxkit.analysis.control.control import ControlValueRequest
from reaxkit.core.analysis_executor import AnalysisExecutor
from reaxkit.core.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.command_alias_resolver import resolve_command_name
from reaxkit.engine.reaxff.generators.control_generator import write_control

CONTROL_COMMANDS = ("get-control",)
MAKE_CONTROL_COMMAND = "make-control"


def _format_value(value):
    """Format numeric values with separators for readable CLI output."""
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        pass

    try:
        fv = float(value)
        if abs(fv) >= 1000:
            int_part, dot, frac = f"{fv}".partition(".")
            int_part = f"{int(int_part):,}"
            return int_part + (("." + frac) if frac else "")
        return value
    except (ValueError, TypeError):
        return value


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None)
    parser.add_argument("--input", default=".", help="Input file or directory for engine resolution")
    parser.add_argument("--run-dir", "--dir", dest="run_dir", default=".", help="Run directory fallback for engine detection")
    parser.add_argument("--control", "--file", dest="control", default="control", help="Path to control file")
    parser.add_argument("--log", choices=["verbose", "quiet"], default=None, help="Logging level")


def _build_get_request(args: argparse.Namespace) -> ControlValueRequest:
    return ControlValueRequest(
        key=args.key,
        section=args.section,
        default=args.default,
    )


REQUEST_BUILDERS: dict[str, Callable[[argparse.Namespace], object]] = {
    "get-control": _build_get_request,
}


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    if command == MAKE_CONTROL_COMMAND:
        parser.set_defaults(command=MAKE_CONTROL_COMMAND)
        parser.formatter_class = argparse.RawTextHelpFormatter
        parser.description = (
            "Generate a default control file template.\n\n"
            "Examples:\n"
            "  reaxkit make-control\n"
            "  reaxkit make-control --output control\n"
            "  reaxkit make-control --output inputs/control"
        )
        parser.add_argument(
            "--output",
            default="reaxkit_generated_inputs/control",
            help="Output path for the generated control file",
        )
        return parser

    canonical = resolve_command_name(command, task_names=CONTROL_COMMANDS)
    parser.set_defaults(command=canonical)
    parser.formatter_class = argparse.RawTextHelpFormatter

    if canonical == "get-control":
        _add_runtime_arguments(parser)
        parser.description = (
            "Get the value of a control parameter.\n\n"
            "Examples:\n"
            "  reaxkit get-control nmdit\n"
            "  reaxkit get-control iout2 --control control --section md\n"
            "  reaxkit get-control imetho --control runs/job1/control"
        )
        parser.add_argument("key", help="Control key to look up, e.g. 'nmdit'")
        parser.add_argument("--section", default=None, help="Optional section: general, md, mm, ff, outdated")
        parser.add_argument("--default", default=None, help="Fallback value when the key is missing")
    else:
        raise KeyError(f"Unsupported control command '{canonical}'.")

    return parser


def _run_get(args: argparse.Namespace) -> int:
    executor = AnalysisExecutor()
    task_cls = TASK_REGISTRY["control_value"]
    request = REQUEST_BUILDERS["get-control"](args)
    result = executor.run(task_cls(), request, vars(args))

    if not result.found and args.default is None:
        print(f"Key '{args.key}' not found in control file '{args.control}'.")
        return 1

    print(f"{result.key} = {_format_value(result.value)}")
    return 0


def _run_make(args: argparse.Namespace) -> int:
    output = write_control(args.output)
    print(f"control file written to {output}")
    return 0


def run_main(command: str, args: argparse.Namespace) -> int:
    if command == MAKE_CONTROL_COMMAND:
        return _run_make(args)

    canonical = resolve_command_name(command, task_names=CONTROL_COMMANDS)
    if canonical == "get-control":
        return _run_get(args)
    raise KeyError(f"Unsupported control command '{canonical}'.")


def _legacy_command_runner(command: str):
    def _runner(args: argparse.Namespace) -> int:
        return run_main(command, args)

    return _runner


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    for command in CONTROL_COMMANDS:
        parser = subparsers.add_parser(command, formatter_class=argparse.RawTextHelpFormatter)
        build_parser(parser, command=command)
        parser.set_defaults(_run=_legacy_command_runner(command))
