"""Template command workflow for direct CLI-driven analysis execution.

This module demonstrates the current ReaxKit workflow structure used by command
families: argument parser construction, request object building, analyzer task
execution, and presentation/export handoff.

**Usage context**

- Command routing: Resolve canonical command names and CLI aliases.
- Task execution: Build request payloads and execute registered analysis tasks.
- Output handling: Present tables/plots and optionally export generated results.

Notes
-----
Replace placeholder request builders and parser options with domain-specific
arguments for your workflow family.
"""

from __future__ import annotations

import argparse
from typing import Callable

import reaxkit.engine  # noqa: F401

from reaxkit.analysis import trajectory as _trajectory_tasks  # noqa: F401
from reaxkit.analysis.trajectory.msd import MSDRequest
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.resolve.command_alias_resolver import resolve_command_name
from reaxkit.core.storage.storage_layout import add_storage_cli_arguments
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.presentation.dispatcher import present_result

ALL_COMMANDS = ("template_metric",)
ALL_LEGACY_COMMANDS = ("template-metric",)


def _build_template_request(args: argparse.Namespace) -> MSDRequest:
    """Build a request object from parsed CLI arguments.

    This helper maps command-line values into the request dataclass expected by
    the selected analysis task.

    Parameters
    -----
    args : argparse.Namespace
        Parsed CLI namespace containing workflow command options.

    Returns
    -----
    MSDRequest
        Request payload passed to the analyzer task.

    Examples
    -----
    ```python
    req = _build_template_request(args)
    ```
    Sample output:
    `MSDRequest(...)`
    Meaning:
    CLI arguments are normalized into a structured task request.
    """
    frames = None
    if args.frames:
        frames = [int(token) for token in args.frames]
    dims = tuple(args.dims) if args.dims else ("x", "y", "z")
    return MSDRequest(
        atom_ids=[int(v) for v in args.atom_ids] if args.atom_ids else None,
        atom_types=list(args.atom_types) if args.atom_types else None,
        dims=dims,
        origin="first",
        frames=frames,
        every=int(args.every),
        unwrap=bool(args.unwrap),
    )


REQUEST_BUILDERS: dict[str, Callable[[argparse.Namespace], object]] = {
    "template_metric": _build_template_request,
}


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Configure and return parser for a workflow command.

    This function applies canonical command resolution and attaches the
    command-specific CLI arguments required to build and run analysis requests.

    Parameters
    -----
    parser : argparse.ArgumentParser
        Parser instance to configure for the selected command.
    command : str
        Command token provided by the dispatcher.

    Returns
    -----
    argparse.ArgumentParser
        Configured parser ready for CLI parsing.

    Examples
    -----
    ```python
    parser = argparse.ArgumentParser()
    parser = build_parser(parser, command="template_metric")
    ```
    Sample output:
    `ArgumentParser(...)`
    Meaning:
    The parser now includes workflow-specific options for request construction.
    """
    canonical = resolve_command_name(command, task_names=ALL_COMMANDS)
    parser.set_defaults(command=canonical)
    parser.set_defaults(progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter

    parser.description = (
        "Run a template analysis command using ReaxKit workflow orchestration.\n"
        "This command demonstrates parser wiring, request construction, task execution,\n"
        "and presentation/export dispatch.\n\n"
        "Examples:\n"
        "  1. Run with defaults:\n"
        "   reaxkit template_metric\n\n"
        "  2. Select atom ids and frames:\n"
        "   reaxkit template_metric --atom-ids 1 2 3 --frames 0 10 20 --every 1\n\n"
        "  3. Filter by atom types and dimensions:\n"
        "   reaxkit template_metric --atom-types O H --dims x y --unwrap"
    )

    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None, help="Engine override for adapter resolution.")
    parser.add_argument("--run-dir", "--dir", dest="run_dir", default=".", help="Run directory fallback for input detection.")
    parser.add_argument("--atom-ids", nargs="*", type=int, default=None, help="Optional atom-id filter list.")
    parser.add_argument("--atom-types", nargs="*", default=None, help="Optional atom-type/element filter list.")
    parser.add_argument("--frames", nargs="*", default=None, help="Optional frame indices to evaluate (space-separated ints).")
    parser.add_argument("--every", type=int, default=1, help="Stride over selected frames.")
    parser.add_argument("--dims", nargs="*", choices=["x", "y", "z"], default=["x", "y", "z"], help="Coordinate dimensions to include.")
    parser.add_argument("--unwrap", action="store_true", help="Enable periodic unwrapping before displacement evaluation.")
    parser.add_argument("--plot", choices=["single", "subplot"], default=None, help="Optional plotting mode.")
    parser.add_argument("--save", default=None, help="Optional output image path.")
    parser.add_argument("--export", default=None, help="Optional CSV export path.")
    add_storage_cli_arguments(parser)
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    """Execute a direct workflow command end-to-end.

    Resolves command name, builds request payload, executes the task through the
    analysis executor, and routes result presentation/export behavior.

    Parameters
    -----
    command : str
        Command token received from CLI dispatch.
    args : argparse.Namespace
        Parsed CLI namespace produced by `build_parser`.

    Returns
    -----
    int
        Process exit code (`0` on success).

    Examples
    -----
    ```python
    status = run_main("template_metric", args)
    ```
    Sample output:
    `0`
    Meaning:
    Command completed successfully and outputs were presented/exported.
    """
    canonical = resolve_command_name(command, task_names=ALL_COMMANDS)
    task_cls = TASK_REGISTRY[canonical]
    request = REQUEST_BUILDERS[canonical](args)

    executor = AnalysisExecutor()
    result = executor.run(task_cls(), request, vars(args))
    present_result(canonical, result, args)
    return 0
