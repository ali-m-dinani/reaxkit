"""Task-oriented workflow for control-file analyses and utilities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis import control as _control_tasks  # noqa: F401
from reaxkit.analysis.control.control import ControlParametersTaskRequest
from reaxkit.core.analysis_executor import AnalysisExecutor
from reaxkit.core.engine_registry import resolve_engine
from reaxkit.core.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.command_alias_resolver import resolve_command_name
from reaxkit.core.generator_runtime import (
    maybe_copy_output_to_dot,
    persist_generator_metadata,
    prepare_generator_output,
    print_saved_dirs,
)
from reaxkit.core.storage_layout import add_storage_cli_arguments
from reaxkit.engine.reaxff.generators.control_generator import gen_control
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.data_models import ControlParametersData

CONTROL_COMMANDS = ("get-control", "write-control")
MAKE_CONTROL_COMMAND = "gen_control"
WRITE_CONTROL_COMMAND = "write-control"


@dataclass
class _LoadControlParametersRequest(BaseRequest):
    """Empty request used to load ``ControlParametersData`` via ``AnalysisExecutor``."""


class _LoadControlParametersTask(AnalysisTask):
    """Pass-through task that returns loaded ``ControlParametersData`` unchanged."""

    required_data = ControlParametersData

    def run(
        self,
        data: ControlParametersData,
        request: _LoadControlParametersRequest,
        reporter=None,
    ) -> ControlParametersData:
        _ = (request, reporter)
        return data


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
    add_storage_cli_arguments(parser)


def _build_get_request(args: argparse.Namespace) -> ControlParametersTaskRequest:
    return ControlParametersTaskRequest(
        key=args.key,
        section=args.section,
    )


REQUEST_BUILDERS: dict[str, Callable[[argparse.Namespace], object]] = {
    "get-control": _build_get_request,
}


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    if command == MAKE_CONTROL_COMMAND:
        parser.set_defaults(command=MAKE_CONTROL_COMMAND)
        parser.set_defaults(progress=True)
        parser.formatter_class = argparse.RawTextHelpFormatter
        parser.description = (
            "Generate a default control file template.\n\n"
            "Examples:\n"
            "  reaxkit gen_control\n"
            "  reaxkit gen_control --output control\n"
            "  reaxkit gen_control --parameter nmdit --value 100000\n"
            "  reaxkit gen_control --output control --copy-to-dot"
        )
        add_storage_cli_arguments(parser)
        parser.add_argument(
            "--output",
            default="control",
            help="Output filename to write under <project_root>/input/",
        )
        parser.add_argument(
            "--copy-to-dot",
            action="store_true",
            help="Also copy the generated control file to the current directory.",
        )
        parser.add_argument(
            "--parameter",
            action="append",
            default=[],
            help="Control parameter key to override (repeatable, must pair with --value).",
        )
        parser.add_argument(
            "--value",
            action="append",
            default=[],
            help="Override value for the corresponding --parameter entry.",
        )
        return parser

    if command == WRITE_CONTROL_COMMAND:
        parser.set_defaults(command=WRITE_CONTROL_COMMAND)
        parser.set_defaults(progress=True)
        parser.formatter_class = argparse.RawTextHelpFormatter
        parser.description = (
            "Read a control file, apply key changes, and write a new control file.\n\n"
            "Examples:\n"
            "  reaxkit write-control --control control --output control.new\n"
            "  reaxkit write-control --control control --parameter nmdit --value 200000 --output control.fast"
        )
        add_storage_cli_arguments(parser)
        parser.add_argument("--engine", choices=["reaxff"], default="reaxff")
        parser.add_argument("--input", default=".", help="Input file or directory for engine resolution")
        parser.add_argument("--run-dir", "--dir", dest="run_dir", default=".", help="Run directory fallback for engine detection")
        parser.add_argument("--control", "--file", dest="control", default="control", help="Path to source control file")
        parser.add_argument(
            "--output",
            default="control",
            help="Output filename to write under <project_root>/input/",
        )
        parser.add_argument(
            "--copy-to-dot",
            action="store_true",
            help="Also copy the generated control file to the current directory.",
        )
        parser.add_argument(
            "--parameter",
            action="append",
            default=[],
            help="Control parameter key to override (repeatable, must pair with --value).",
        )
        parser.add_argument(
            "--value",
            action="append",
            default=[],
            help="Override value for the corresponding --parameter entry.",
        )
        parser.add_argument("--log", choices=["verbose", "quiet"], default=None, help="Logging level")
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
    row = result.table.iloc[0] if getattr(result, "table", None) is not None and not result.table.empty else None
    found = bool(row["found"]) if row is not None and "found" in row else False
    key = str(row["key"]) if row is not None and "key" in row else str(args.key)
    value = row["value"] if row is not None and "value" in row else args.default

    if not found and args.default is None:
        print(f"Key '{args.key}' not found in control file '{args.control}'.")
        return 1

    print(f"{key} = {_format_value(value)}")
    return 0


def _run_make(args: argparse.Namespace) -> int:
    output, layout = prepare_generator_output(
        args,
        command=MAKE_CONTROL_COMMAND,
        output_value=str(getattr(args, "output", "control")),
    )
    parameters = list(getattr(args, "parameter", []) or [])
    values = list(getattr(args, "value", []) or [])
    if len(parameters) != len(values):
        raise ValueError("--parameter and --value must be provided the same number of times.")
    overrides = {str(k).strip(): str(v) for k, v in zip(parameters, values)}

    gen_control(output, overrides=overrides or None)
    persist_generator_metadata(
        args,
        command=MAKE_CONTROL_COMMAND,
        output_path=output,
        layout=layout,
        extra={"overrides": overrides},
        copy_to_dot=bool(getattr(args, "copy_to_dot", False)),
    )
    copied = maybe_copy_output_to_dot(output, enabled=bool(getattr(args, "copy_to_dot", False)))
    dirs = [output.parent]
    if copied is not None:
        dirs.append(copied.parent)
    print_saved_dirs(dirs)
    return 0


def _run_write(args: argparse.Namespace) -> int:
    output, layout = prepare_generator_output(
        args,
        command=WRITE_CONTROL_COMMAND,
        output_value=str(getattr(args, "output", "control")),
    )
    parameters = list(getattr(args, "parameter", []) or [])
    values = list(getattr(args, "value", []) or [])
    if len(parameters) != len(values):
        raise ValueError("--parameter and --value must be provided the same number of times.")
    overrides = {str(k).strip(): str(v) for k, v in zip(parameters, values)}

    executor = AnalysisExecutor()
    data = executor.run(
        _LoadControlParametersTask(),
        _LoadControlParametersRequest(),
        vars(args),
    )

    source = str(getattr(args, "control", "control"))
    adapter = resolve_engine(source, engine=getattr(args, "engine", None))
    adapter.write(
        data,
        output,
        args={"overrides": overrides or None},
    )

    persist_generator_metadata(
        args,
        command=WRITE_CONTROL_COMMAND,
        output_path=output,
        layout=layout,
        extra={"source_control": source, "overrides": overrides},
        copy_to_dot=bool(getattr(args, "copy_to_dot", False)),
    )
    copied = maybe_copy_output_to_dot(output, enabled=bool(getattr(args, "copy_to_dot", False)))
    dirs = [output.parent]
    if copied is not None:
        dirs.append(copied.parent)
    print_saved_dirs(dirs)
    return 0


def run_main(command: str, args: argparse.Namespace) -> int:
    if command == MAKE_CONTROL_COMMAND:
        return _run_make(args)
    if command == WRITE_CONTROL_COMMAND:
        return _run_write(args)

    canonical = resolve_command_name(command, task_names=CONTROL_COMMANDS)
    if canonical == "get-control":
        return _run_get(args)
    if canonical == "write-control":
        return _run_write(args)
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
