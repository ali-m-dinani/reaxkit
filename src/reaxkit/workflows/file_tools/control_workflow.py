"""Task-oriented workflow for control-file analyses and utilities.

This module implements CLI workflow orchestration for its command family, including argument parsing, request construction, execution dispatch, and result presentation handoff.

**Usage context**

- Command routing: Resolve CLI aliases and normalized command names.
- Task execution: Build request objects and invoke registered tasks.
- Output handling: Forward results to table, plot, export, or report flows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis import control as _control_tasks  # noqa: F401
from reaxkit.analysis.control.control import ControlParametersTaskRequest
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.core.platform.engine_resolver import resolve_engine
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.resolve.command_alias_resolver import resolve_command_name
from reaxkit.core.runtime.generator_runtime import (
    maybe_copy_output_to_dot,
    persist_generator_metadata,
    prepare_generator_output,
    print_saved_dirs,
)
from reaxkit.core.storage.storage_layout import add_storage_cli_arguments
from reaxkit.engine.reaxff.generators.control_generator import gen_control
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.data_models import ControlParametersData

ALL_COMMANDS = ("get-control_data", "gen-control", "gen_template_control")
ALL_LEGACY_COMMANDS = ("get_control_data", "gen_control", "write-control", "write_control", "make-control", "make_control")
MAKE_CONTROL_COMMAND = "gen_template_control"
WRITE_CONTROL_COMMAND = "gen-control"


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
        """Run.

        Execute the workflow function for this command path and return the
        computed result for downstream CLI handling.

        Parameters
        -----
        data : Any
            Function argument.
        request : Any
            Function argument.
        reporter : Any
            Function argument.

        Returns
        -----
        ControlParametersData
            Function return value.

        Examples
        -----
        >>> # See workflow CLI usage for concrete examples.
        """
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
    """Add runtime arguments."""
    parser.add_argument(
        "--engine",
        choices=["reaxff", "ams", "lammps"],
        default=None,
        help="Engine override. Example: --engine reaxff, which forces ReaxFF parsing/writing rules.",
    )
    parser.add_argument(
        "--input",
        default=".",
        help="Input file or directory for engine resolution. Example: --input runs/job1, which tells the resolver where to inspect files.",
    )
    parser.add_argument(
        "--run-dir",
        "--dir",
        dest="run_dir",
        default=".",
        help="Run directory fallback for engine detection. Example: --run-dir runs/job1, which is used when --input is not enough to resolve context.",
    )
    parser.add_argument(
        "--control",
        "--file",
        dest="control",
        default="control",
        help="Path to control file. Example: --control runs/job1/control, which reads that specific control file instead of the default one.",
    )
    parser.add_argument(
        "--log",
        choices=["verbose", "quiet"],
        default=None,
        help="Logging level. Example: --log verbose, which prints more runtime details.",
    )
    add_storage_cli_arguments(parser)


def _build_get_request(args: argparse.Namespace) -> ControlParametersTaskRequest:
    """Build get request."""
    return ControlParametersTaskRequest(
        key=args.key,
        section=args.section,
    )


REQUEST_BUILDERS: dict[str, Callable[[argparse.Namespace], object]] = {
    "get-control_data": _build_get_request,
}


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Build parser.

    Execute the workflow function for this command path and return the
    computed result for downstream CLI handling.

    Parameters
    -----
    parser : Any
        Function argument.
    command : Any
        Function argument.

    Returns
    -----
    argparse.ArgumentParser
        Function return value.

    Examples
    -----
    >>> # See workflow CLI usage for concrete examples.
    """
    if command == MAKE_CONTROL_COMMAND:
        parser.set_defaults(command=MAKE_CONTROL_COMMAND)
        parser.set_defaults(progress=True)
        parser.formatter_class = argparse.RawTextHelpFormatter
        parser.description = (
            "Write a template ReaxFF control file.\n"
            "This command generates a starter control file only. It does not run a simulation.\n"
            "You can optionally override one or more parameters at generation time by repeating\n"
            "--parameter/--value pairs.\n\n"
            "Examples:\n"
            "  1. Generate a default control template ('control'):\n"
            "   reaxkit gen_template_control\n\n"
            "  2. Generate a template and override one parameter:\n"
            "   reaxkit gen_template_control --parameter nmdit --value 100000\n\n"
            "  3. Generate a template and also copy it to the current directory:\n"
            "   reaxkit gen_template_control --output control --copy-to-dot"
        )
        add_storage_cli_arguments(parser)
        parser.add_argument(
            "--output",
            default="control",
            help="Output filename under <project_root>/input/. Example: --output control.fast, which writes the generated template with that filename.",
        )
        parser.add_argument(
            "--copy-to-dot",
            action="store_true",
            help="Also copy the generated control file to the current directory. Example: --copy-to-dot, which keeps an extra copy beside where you run the command.",
        )
        parser.add_argument(
            "--parameter",
            action="append",
            default=[],
            help="Control parameter key to override (repeatable, pair with --value). Example: --parameter nmdit, which selects the key to change.",
        )
        parser.add_argument(
            "--value",
            action="append",
            default=[],
            help="Override value for the corresponding --parameter entry. Example: --value 100000, which sets the new value for the paired key.",
        )
        return parser

    if command == WRITE_CONTROL_COMMAND:
        parser.set_defaults(command=WRITE_CONTROL_COMMAND)
        parser.set_defaults(progress=True)
        parser.formatter_class = argparse.RawTextHelpFormatter
        parser.description = (
            "Read an existing control file, apply parameter overrides, and write an updated file.\n"
            "Use this command when you want to keep most of a control file unchanged while updating\n"
            "specific keys through repeatable --parameter/--value pairs.\n\n"
            "Examples:\n"
            "  1. Copy a control file to a new output name without changing parameters:\n"
            "   reaxkit gen-control --control control --output control.new\n\n"
            "  2. Update one parameter while writing a new control file:\n"
            "   reaxkit gen-control --control control --parameter nmdit --value 200000 --output control.fast"
        )
        add_storage_cli_arguments(parser)
        parser.add_argument(
            "--engine",
            choices=["reaxff"],
            default="reaxff",
            help="Engine type for control IO. Example: --engine reaxff, which applies ReaxFF-specific control-file handling.",
        )
        parser.add_argument(
            "--input",
            default=".",
            help="Input file or directory for engine resolution. Example: --input runs/job1, which points resolution to that run location.",
        )
        parser.add_argument(
            "--run-dir",
            "--dir",
            dest="run_dir",
            default=".",
            help="Run directory fallback for engine detection. Example: --run-dir runs/job1, which acts as backup context for engine detection.",
        )
        parser.add_argument(
            "--control",
            "--file",
            dest="control",
            default="control",
            help="Path to source control file. Example: --control runs/job1/control, which is the file to read and modify.",
        )
        parser.add_argument(
            "--output",
            default="control",
            help="Output filename under <project_root>/input/. Example: --output control.new, which writes the updated file under that name.",
        )
        parser.add_argument(
            "--copy-to-dot",
            action="store_true",
            help="Also copy the generated control file to the current directory. Example: --copy-to-dot, which creates a convenience copy in your working directory.",
        )
        parser.add_argument(
            "--parameter",
            action="append",
            default=[],
            help="Control parameter key to override (repeatable, pair with --value). Example: --parameter nmdit, which marks `nmdit` for replacement.",
        )
        parser.add_argument(
            "--value",
            action="append",
            default=[],
            help="Override value for the corresponding --parameter entry. Example: --value 200000, which becomes the new value for the matched parameter key.",
        )
        parser.add_argument(
            "--log",
            choices=["verbose", "quiet"],
            default=None,
            help="Logging level. Example: --log quiet, which suppresses non-essential log output.",
        )
        return parser

    canonical = resolve_command_name(command, task_names=ALL_COMMANDS)
    parser.set_defaults(command=canonical)
    parser.formatter_class = argparse.RawTextHelpFormatter

    if canonical == "get-control_data":
        _add_runtime_arguments(parser)
        parser.description = (
            "Read and print the value of one control parameter key.\n"
            "Use this command to quickly inspect a control file without opening or parsing it manually.\n"
            "You can optionally scope lookup to a section and provide a fallback default value.\n\n"
            "Examples:\n"
            "  1. Read a key from the default control file ('control'):\n"
            "   reaxkit get-control_data nmdit\n\n"
            "  2. Read a key from a specific section:\n"
            "   reaxkit get-control_data iout2 --control control --section md\n\n"
            "  3. Read a key from a control file at a custom path:\n"
            "   reaxkit get-control_data imetho --control runs/job1/control"
        )
        parser.add_argument("key", help="Control key to look up. Example: nmdit, which queries the `nmdit` parameter value.")
        parser.add_argument(
            "--section",
            default=None,
            help="Optional section: general, md, mm, ff, outdated. Example: --section md, which narrows lookup to the MD section.",
        )
        parser.add_argument(
            "--default",
            default=None,
            help="Fallback value if key is missing. Example: --default 0, which prints 0 instead of failing when the key is absent.",
        )
    else:
        raise KeyError(f"Unsupported control command '{canonical}'.")

    return parser


def _run_get(args: argparse.Namespace) -> int:
    """Run get."""
    executor = AnalysisExecutor()
    task_cls = TASK_REGISTRY["get_control_data"]
    request = REQUEST_BUILDERS["get-control_data"](args)
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
    """Run make."""
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
    """Run write."""
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
    """Run main.

    Execute the workflow function for this command path and return the
    computed result for downstream CLI handling.

    Parameters
    -----
    command : Any
        Function argument.
    args : Any
        Function argument.

    Returns
    -----
    int
        Function return value.

    Examples
    -----
    >>> # See workflow CLI usage for concrete examples.
    """
    if command == MAKE_CONTROL_COMMAND:
        return _run_make(args)
    if command == WRITE_CONTROL_COMMAND:
        return _run_write(args)

    canonical = resolve_command_name(command, task_names=ALL_COMMANDS)
    if canonical == "get-control_data":
        return _run_get(args)
    if canonical == "gen-control":
        return _run_write(args)
    raise KeyError(f"Unsupported control command '{canonical}'.")


def _legacy_command_runner(command: str):
    """Legacy command runner."""
    def _runner(args: argparse.Namespace) -> int:
        """Runner."""
        return run_main(command, args)

    return _runner


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    """Register tasks.

    Execute the workflow function for this command path and return the
    computed result for downstream CLI handling.

    Parameters
    -----
    subparsers : Any
        Function argument.

    Returns
    -----
    None
        Function return value.

    Examples
    -----
    >>> # See workflow CLI usage for concrete examples.
    """
    for command in ALL_COMMANDS:
        parser = subparsers.add_parser(command, formatter_class=argparse.RawTextHelpFormatter)
        build_parser(parser, command=command)
        parser.set_defaults(_run=_legacy_command_runner(command))
