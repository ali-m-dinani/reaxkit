"""Direct command workflow for optimization-parameter analysis.

This module implements CLI workflow orchestration for its command family, including argument parsing, request construction, execution dispatch, and result presentation handoff.

**Usage context**

- Command routing: Resolve CLI aliases and normalized command names.
- Task execution: Build request objects and invoke registered tasks.
- Output handling: Forward results to table, plot, export, or report flows.
"""

from __future__ import annotations

import argparse
from typing import Callable

from reaxkit.analysis import params as _params_tasks  # noqa: F401
from reaxkit.analysis.params.params import ForceFieldOptimizationParameterRequest
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.resolve.command_alias_resolver import resolve_command_name
from reaxkit.core.storage.storage_layout import add_storage_cli_arguments
from reaxkit.presentation.dispatcher import present_result

ALL_COMMANDS = ("get-params",)
ALL_LEGACY_COMMANDS = ("get_params",)


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Add runtime arguments."""
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None, help="Engine override. Example: --engine reaxff, which applies ReaxFF parsing/loading behavior.")
    parser.add_argument("--input", default=".", help="Input file or directory for engine resolution. Example: --input runs/job1, which sets the lookup context for files.")
    parser.add_argument("--run-dir", "--dir", dest="run_dir", default=".", help="Run directory fallback for engine detection. Example: --run-dir runs/job1, which acts as backup resolution path.")
    parser.add_argument("--params", "--file", dest="params", default="params", help="Path to params file. Example: --params params, which reads optimization parameter definitions from that file.")
    parser.add_argument("--log", choices=["verbose", "quiet"], default=None, help="Logging level. Example: --log verbose, which prints more runtime details.")
    add_storage_cli_arguments(parser)


def _add_presentation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add presentation arguments."""
    parser.add_argument("--plot", choices=["single", "subplot"], default=None, help="Render a plot. Example: --plot single, which draws one combined chart.")
    parser.add_argument("--show", action="store_true", help="Show the generated plot window. Example: --show, which opens the plot interactively.")
    parser.add_argument("--save", default=None, help="Save the generated plot to a file path. Example: --save params.png, which writes the figure image.")
    parser.add_argument("--export", default=None, help="Write the result table to CSV. Example: --export params.csv, which saves tabular output.")
    parser.add_argument("--grid", default=None, help="Subplot grid like 2x2 or 2*2. Example: --grid 2x2, which arranges subplot panels in a 2-by-2 layout.")
    parser.add_argument(
        "--xaxis",
        choices=["ff_section_line", "ff_parameter"],
        default="ff_section_line",
        help="Quantity on x-axis. Example: --xaxis ff_parameter, which uses parameter-id/name style values on the horizontal axis.",
    )


def _build_params_request(args: argparse.Namespace) -> ForceFieldOptimizationParameterRequest:
    """Build params request."""
    return ForceFieldOptimizationParameterRequest(
        drop_duplicate=not args.keep_duplicates,
        interpret=args.interpret,
    )


REQUEST_BUILDERS: dict[str, Callable[[argparse.Namespace], object]] = {
    "get-params": _build_params_request,
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
    canonical = resolve_command_name(command, task_names=ALL_COMMANDS)
    parser.set_defaults(command=canonical)
    parser.set_defaults(progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter

    _add_runtime_arguments(parser)
    _add_presentation_arguments(parser)

    if canonical == "get-params":
        parser.description = (
            "Load optimization-parameter definitions from `params`.\n"
            "This command can return raw parameter rows or interpreted mappings into `ffield`\n"
            "when pointer interpretation is enabled.\n\n"
            "Examples:\n"
            "  1. Export raw parameters:\n"
            "   reaxkit get-params --export params.csv\n\n"
            "  2. Interpret parameter pointers using force-field file and export:\n"
            "   reaxkit get-params --interpret --ffield ffield --export params_interpreted.csv\n\n"
            "  3. Plot parameter search intervals:\n"
            "   reaxkit get-params --plot single"
        )
        parser.add_argument("--keep-duplicates", action="store_true", help="Do not drop duplicate parameter rows. Example: --keep-duplicates, which keeps repeated rows exactly as found in input.")
        parser.add_argument("--interpret", action="store_true", help="Interpret params pointers into the ffield. Example: --interpret, which resolves pointer-style entries to force-field context.")
        parser.add_argument("--ffield", default=None, help="Path to ffield file required for --interpret. Example: --ffield ffield, which provides force-field content used for pointer interpretation.")
    else:
        raise KeyError(f"Unsupported params command '{canonical}'.")

    return parser


def _plot_payload(command: str, result, args: argparse.Namespace) -> dict[str, object] | None:
    """Plot payload."""
    table = result.table
    if table.empty or command != "get-params":
        return None

    x_col = getattr(args, "xaxis", "ff_section_line")
    if x_col not in table.columns:
        return None

    y_col = "search_interval"
    if y_col not in table.columns:
        return None

    if getattr(args, "plot", None) == "subplot" and "ffield_section_name" in table.columns:
        subplots = []
        for section, group in table.groupby("ffield_section_name", sort=True):
            subplots.append(
                [{
                    "x": group[x_col].tolist(),
                    "y": group[y_col].tolist(),
                    "label": str(section),
                }]
            )
        return {
            "plot_type": "multi_subplots",
            "subplots": subplots,
            "xlabel": x_col,
            "ylabel": y_col,
            "title": "Optimization Parameters",
            "legend": False,
            "grid": getattr(args, "grid", None),
        }

    if "ffield_section_name" in table.columns:
        series = []
        for section, group in table.groupby("ffield_section_name", sort=True):
            series.append(
                {
                    "x": group[x_col].tolist(),
                    "y": group[y_col].tolist(),
                    "label": str(section),
                }
            )
        return {
            "plot_type": "single_plot",
            "series": series,
            "xlabel": x_col,
            "ylabel": y_col,
            "title": "Optimization Parameters",
            "legend": True,
        }

    return {
        "plot_type": "single_plot",
        "x": table[x_col].tolist(),
        "y": table[y_col].tolist(),
        "xlabel": x_col,
        "ylabel": y_col,
        "title": "Optimization Parameters",
    }


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
    canonical = resolve_command_name(command, task_names=ALL_COMMANDS)
    task_cls = TASK_REGISTRY["force_field_optimization_parameters"]
    request = REQUEST_BUILDERS[canonical](args)

    executor = AnalysisExecutor()
    result = executor.run(task_cls(), request, vars(args))
    present_result(canonical, result, args, plot_payload_builder=_plot_payload)
    return 0
