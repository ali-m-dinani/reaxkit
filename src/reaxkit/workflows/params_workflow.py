"""Direct command workflow for optimization-parameter analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from reaxkit.analysis import params as _params_tasks  # noqa: F401
from reaxkit.analysis.params.params import ForceFieldOptimizationParameterRequest
from reaxkit.core.analysis_executor import AnalysisExecutor
from reaxkit.core.engine_registry import resolve_engine
from reaxkit.core.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.command_alias_resolver import resolve_command_name
from reaxkit.domain.data_models import ForceFieldParametersData
from reaxkit.presentation.dispatcher import present_result

PARAMS_COMMANDS = ("get-params",)


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None)
    parser.add_argument("--input", default=".", help="Input file or directory for engine resolution")
    parser.add_argument("--run-dir", "--dir", dest="run_dir", default=".", help="Run directory fallback for engine detection")
    parser.add_argument("--params", "--file", dest="params", default="params", help="Path to params file")
    parser.add_argument("--log", choices=["verbose", "quiet"], default=None, help="Logging level")


def _add_presentation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plot", choices=["single", "subplot"], default=None, help="Render a plot")
    parser.add_argument("--show", action="store_true", help="Show the generated plot window")
    parser.add_argument("--save", default=None, help="Save the generated plot to a file path")
    parser.add_argument("--export", default=None, help="Write the result table to CSV")
    parser.add_argument("--grid", default=None, help="Subplot grid like 2x2 or 2*2")
    parser.add_argument(
        "--xaxis",
        choices=["ff_section_line", "ff_parameter"],
        default="ff_section_line",
        help="Quantity on x-axis",
    )


def _maybe_load_force_field(args: argparse.Namespace) -> ForceFieldParametersData | None:
    if not getattr(args, "interpret", False):
        return None
    raw = getattr(args, "ffield", None)
    if not raw:
        raise ValueError("--ffield is required when --interpret is set.")
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(f"ffield file not found: {raw}")
    adapter = resolve_engine(str(path), engine=getattr(args, "engine", None))
    return adapter.load(ForceFieldParametersData, vars(args))


def _build_params_request(args: argparse.Namespace) -> ForceFieldOptimizationParameterRequest:
    return ForceFieldOptimizationParameterRequest(
        sort_by=args.sort_by,
        ascending=not args.descending,
        drop_duplicate=not args.keep_duplicates,
        interpret=args.interpret,
        force_field=_maybe_load_force_field(args),
        add_term=not args.no_term,
        sep=args.sep,
    )


REQUEST_BUILDERS: dict[str, Callable[[argparse.Namespace], object]] = {
    "get-params": _build_params_request,
}


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    canonical = resolve_command_name(command, task_names=PARAMS_COMMANDS)
    parser.set_defaults(command=canonical)
    parser.formatter_class = argparse.RawTextHelpFormatter

    _add_runtime_arguments(parser)
    _add_presentation_arguments(parser)

    if canonical == "get-params":
        parser.description = (
            "Load raw or interpreted optimization-parameter definitions from params.\n\n"
            "Examples:\n"
            "  reaxkit get-params --export params.csv\n"
            "  reaxkit get-params --interpret --ffield ffield --export params_interpreted.csv\n"
            "  reaxkit get-params --sort-by search_interval --descending --plot single"
        )
        parser.add_argument("--keep-duplicates", action="store_true", help="Do not drop duplicate parameter rows")
        parser.add_argument("--sort-by", default=None, help="Optional column name to sort by")
        parser.add_argument("--descending", action="store_true", help="Sort in descending order when --sort-by is used")
        parser.add_argument("--interpret", action="store_true", help="Interpret params pointers into the ffield")
        parser.add_argument("--ffield", default=None, help="Path to ffield file required for --interpret")
        parser.add_argument("--no-term", action="store_true", help="Do not build readable term labels during interpretation")
        parser.add_argument("--sep", default="-", help="Separator used when constructing interpreted term labels")
    else:
        raise KeyError(f"Unsupported params command '{canonical}'.")

    return parser


def _plot_payload(command: str, result, args: argparse.Namespace) -> dict[str, object] | None:
    table = result.table
    if table.empty or command != "get-params":
        return None

    x_col = getattr(args, "xaxis", "ff_section_line")
    if x_col not in table.columns:
        return None

    y_col = "search_interval"
    if y_col not in table.columns:
        return None

    if getattr(args, "plot", None) == "subplot" and "ffield_section_key" in table.columns:
        subplots = []
        for section, group in table.groupby("ffield_section_key", sort=True):
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

    if "ffield_section_key" in table.columns:
        series = []
        for section, group in table.groupby("ffield_section_key", sort=True):
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
    canonical = resolve_command_name(command, task_names=PARAMS_COMMANDS)
    task_cls = TASK_REGISTRY["force_field_optimization_parameters"]
    request = REQUEST_BUILDERS[canonical](args)

    executor = AnalysisExecutor()
    result = executor.run(task_cls(), request, vars(args))
    present_result(canonical, result, args, plot_payload_builder=_plot_payload)
    return 0
