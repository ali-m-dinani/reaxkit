"""Direct command workflow for active-site analyses.

This module implements CLI workflow orchestration for its command family, including argument parsing, request construction, execution dispatch, and result presentation handoff.

**Usage context**

- Command routing: Resolve CLI aliases and normalized command names.
- Task execution: Build request objects and invoke registered tasks.
- Output handling: Forward results to table, plot, export, or report flows.
"""

from __future__ import annotations

import argparse
from typing import Callable

import pandas as pd

from reaxkit.analysis import active_sites as _active_site_tasks  # noqa: F401
from reaxkit.analysis.active_sites import (
    ActiveSiteEventsRequest,
    ActiveSiteStructuralRequest,
)
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.resolve.command_alias_resolver import resolve_command_name
from reaxkit.core.utils.frame_utils import parse_frame_indices
from reaxkit.core.storage.storage_layout import add_storage_cli_arguments
from reaxkit.presentation.dispatcher import present_result
from reaxkit.core.results_shaping.result_bundle import bundle_canonical_and_tract_tables

ALL_COMMANDS = ("get_active_site_structural", "get_active_site_events")
ALL_LEGACY_COMMANDS = (
    "active_site_structural",
    "active_site_events",
    "get-active-site-structural",
    "get-active-site-events",
    "active-site-structural",
    "active-site-events",
)


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Add runtime arguments."""
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None)
    parser.add_argument("--input", default=".", help="Input file or directory for engine resolution")
    parser.add_argument("--run-dir", "--dir", dest="run_dir", default=".", help="Run directory fallback for engine detection")
    parser.add_argument("--fort7", default="fort.7", help="Path to fort.7")
    parser.add_argument("--xmolout", default="xmolout", help="Path to xmolout")
    parser.add_argument("--summary", default=None, help="Optional summary.txt path")
    parser.add_argument("--log", choices=["verbose", "quiet"], default=None, help="Logging level")
    add_storage_cli_arguments(parser)


def _add_presentation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add presentation arguments."""
    parser.add_argument("--plot", choices=["single", "subplot"], default=None, help="Render a plot")
    parser.add_argument("--show", action="store_true", help="Show the generated plot window")
    parser.add_argument("--save", default=None, help="Save the generated plot to a file path")
    parser.add_argument("--export", default=None, help="Write the result table to CSV")
    parser.add_argument("--grid", default=None, help="Subplot grid like 2x2 or 2*2")
    parser.add_argument(
        "--report",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate a report under reports/<command>/<analysis_id>/",
    )
    parser.add_argument(
        "--report-format",
        choices=["both", "pdf", "docx"],
        default="both",
        help="Report format when --report is enabled.",
    )


def _build_active_site_structural_request(args: argparse.Namespace) -> ActiveSiteStructuralRequest:
    """Build active site structural request."""
    return ActiveSiteStructuralRequest(
        frame=int(args.frame),
        bo_threshold=float(args.bo_threshold),
        bond_mode=str(args.bond_mode),
        bond_scale=float(args.bond_scale),
        alpha_radius=float(args.alpha_radius),
        gap_deg=float(args.gap_deg),
        carbon_element=str(args.carbon_element),
        include_noncarbon=bool(args.include_noncarbon),
        strict_tract=bool(args.strict_tract),
        soap=bool(args.soap),
        soap_ref_path=args.soap_ref_path,
        soap_r_cut=float(args.soap_r_cut),
        soap_n_max=int(args.soap_n_max),
        soap_l_max=int(args.soap_l_max),
        soap_zeta=int(args.soap_zeta),
    )


def _build_active_site_events_request(args: argparse.Namespace) -> ActiveSiteEventsRequest:
    """Build active site events request."""
    return ActiveSiteEventsRequest(
        frames=parse_frame_indices(args.frames),
        every=int(args.every),
        mode=str(args.mode),
        bo_threshold=float(args.bo_threshold),
        r_CO=float(args.r_co),
        r_CSi=float(args.r_csi),
        persist=int(args.persist),
        carbon_element=str(args.carbon_element),
        oxygen_element=str(args.oxygen_element),
        silicon_element=str(args.silicon_element),
        strict_tract=bool(args.strict_tract),
    )


REQUEST_BUILDERS: dict[str, Callable[[argparse.Namespace], object]] = {
    "get_active_site_structural": _build_active_site_structural_request,
    "get_active_site_events": _build_active_site_events_request,
}

TASK_KEY_BY_COMMAND = {
    "get_active_site_structural": "active_site_structural",
    "get_active_site_events": "active_site_events",
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
    parser.formatter_class = argparse.RawTextHelpFormatter

    _add_runtime_arguments(parser)
    _add_presentation_arguments(parser)

    if canonical == "get_active_site_structural":
        parser.description = (
            "Compute per-atom active-site structural descriptors on a selected frame.\n"
            "Use this command when you need atom-resolved structural labels and geometry metrics\n"
            "around active sites from connectivity trajectory data.\n"
            "This command analyzes prepared analysis inputs and does not run a simulation.\n\n"
            "Examples:\n"
            "  1. Baseline structural analysis on frame 0:\n"
            "   reaxkit get_active_site_structural --frame 0 --bo-threshold 0.3\n\n"
            "  2. Strict carbon-focused analysis on a later frame:\n"
            "   reaxkit get_active_site_structural --frame 100 --no-include-noncarbon --strict-tract\n\n"
            "  3. Render and save a structural plot:\n"
            "   reaxkit get_active_site_structural --plot single --save active_site_structural.png"
        )
        parser.add_argument("--frame", type=int, default=0, help="Frame index used for structural analysis. Example: --frame 100, which runs the descriptor extraction on frame 100.")
        parser.add_argument("--bo-threshold", type=float, default=0.3, help="Bond-order threshold used to build connectivity. Example: --bo-threshold 0.4, which requires stronger bonds to count as connected.")
        parser.add_argument(
            "--bond-mode",
            choices=["bo", "distance"],
            default="bo",
            help="Bond graph source: bo (from ConnectivityData.bond_orders) or distance (TRACT geometric cutoffs)",
        )
        parser.add_argument("--bond-scale", type=float, default=1.20, help="Scale factor on covalent radii for distance mode")
        parser.add_argument("--alpha-radius", type=float, default=0.0, help="Alpha-shape radius for non-periodic boundary detection")
        parser.add_argument("--gap-deg", type=float, default=220.0, help="Angular-gap threshold for boundary fallback")
        parser.add_argument("--carbon-element", default="C", help="Element symbol used for carbon network analysis")
        parser.add_argument(
            "--include-noncarbon",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Include non-carbon atoms in output table",
        )
        parser.add_argument(
            "--strict-tract",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Raise if canonical structural output cannot satisfy strict TRACT compatibility",
        )
        parser.add_argument(
            "--soap",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Compute optional SOAP descriptors (soap_pc1/2/3 and optional soap_score).",
        )
        parser.add_argument("--soap-ref-path", default=None, help="Optional .npy reference SOAP vectors for soap_score.")
        parser.add_argument("--soap-r-cut", type=float, default=5.0, help="SOAP cutoff radius in angstrom.")
        parser.add_argument("--soap-n-max", type=int, default=9, help="SOAP radial basis size.")
        parser.add_argument("--soap-l-max", type=int, default=9, help="SOAP angular basis size.")
        parser.add_argument("--soap-zeta", type=int, default=2, help="SOAP kernel exponent for reference similarity.")
    elif canonical == "get_active_site_events":
        parser.description = (
            "Extract persistent active-site C-O and C-Si events across trajectory frames.\n"
            "Use this command to detect bond-forming or bond-breaking event patterns over time\n"
            "from connectivity-aware data.\n"
            "This command analyzes existing data and does not generate force-field input templates.\n\n"
            "Examples:\n"
            "  1. Automatic mode for persistent events:\n"
            "   reaxkit get_active_site_events --mode auto --persist 5\n\n"
            "  2. Bond-order mode on a sampled frame range:\n"
            "   reaxkit get_active_site_events --frames 0:500:5 --mode bo --bo-threshold 0.8\n\n"
            "  3. Distance mode with custom cutoffs:\n"
            "   reaxkit get_active_site_events --mode dist --r-co 1.65 --r-csi 2.10 --strict-tract"
        )
        parser.add_argument(
            "--frames",
            nargs="*",
            default=None,
            help='Frames: "0,10,20", "0 10 20", "0:20", "0-20", or "0:20:2"',
        )
        parser.add_argument("--every", type=int, default=10, help="Use every Nth selected frame (default: 10)")
        parser.add_argument("--mode", choices=["auto", "bo", "dist"], default="auto", help="Event detection mode")
        parser.add_argument("--bo-threshold", type=float, default=0.8, help="Bond-order threshold for bo mode")
        parser.add_argument("--r-co", type=float, default=1.65, help="C-O distance cutoff in angstrom for dist mode")
        parser.add_argument("--r-csi", type=float, default=2.10, help="C-Si distance cutoff in angstrom for dist mode")
        parser.add_argument("--persist", type=int, default=50, help="Required consecutive analyzed frames for confirmed binding")
        parser.add_argument("--carbon-element", default="C", help="Carbon element symbol")
        parser.add_argument("--oxygen-element", default="O", help="Oxygen element symbol")
        parser.add_argument("--silicon-element", default="Si", help="Silicon element symbol")
        parser.add_argument(
            "--strict-tract",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Raise if canonical events output cannot satisfy strict TRACT compatibility",
        )
    else:
        raise KeyError(f"Unsupported active-site command '{canonical}'.")

    return parser


def _plot_payload(command: str, result, _args: argparse.Namespace) -> dict[str, object] | None:
    """Plot payload."""
    table = result.table
    if not isinstance(table, pd.DataFrame) or table.empty:
        return None

    if command == "get_active_site_structural":
        if {"x", "y", "label"}.issubset(table.columns):
            series = []
            for lbl, group in table.groupby("label", sort=True):
                series.append(
                    {
                        "x": group["x"].tolist(),
                        "y": group["y"].tolist(),
                        "label": str(lbl),
                        "markersize": 8,
                        "alpha": 0.8,
                    }
                )
            return {
                "plot_type": "single_plot",
                "series": series,
                "xlabel": "x [A]",
                "ylabel": "y [A]",
                "title": "Atom Label Map",
                "legend": True,
                "kind": "scatter",
            }
        if not {"atom_id", "d_pyr"}.issubset(table.columns):
            return None
        ordered = table.sort_values("atom_id")
        return {
            "plot_type": "single_plot",
            "x": ordered["atom_id"].tolist(),
            "y": ordered["d_pyr"].tolist(),
            "xlabel": "atom_id",
            "ylabel": "d_pyr",
            "title": "Pyramidalization by Atom",
        }

    if command == "get_active_site_events":
        if not {"atom_id", "n_events_O", "n_events_Si"}.issubset(table.columns):
            return None
        ordered = table.sort_values("atom_id")
        return {
            "plot_type": "single_plot",
            "series": [
                {"x": ordered["atom_id"].tolist(), "y": ordered["n_events_O"].tolist(), "label": "n_events_O"},
                {"x": ordered["atom_id"].tolist(), "y": ordered["n_events_Si"].tolist(), "label": "n_events_Si"},
            ],
            "xlabel": "atom_id",
            "ylabel": "event_count",
            "title": "Active-Site Event Counts",
            "legend": True,
        }

    return None


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
    task_cls = TASK_REGISTRY[TASK_KEY_BY_COMMAND[canonical]]
    request = REQUEST_BUILDERS[canonical](args)

    executor = AnalysisExecutor()
    result = executor.run(task_cls(), request, vars(args))
    bundled = bundle_canonical_and_tract_tables(result)
    present_result(
        canonical,
        bundled,
        args,
        plot_payload_builder=_plot_payload,
    )
    return 0
