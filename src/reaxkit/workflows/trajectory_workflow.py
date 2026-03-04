"""Direct command workflow for trajectory analyses."""

from __future__ import annotations

import argparse
import numpy as np
from typing import Callable

from reaxkit.analysis import trajectory as _trajectory_tasks  # noqa: F401
from reaxkit.analysis.trajectory.msd import MSDRequest
from reaxkit.analysis.trajectory.rdf import RDFPropertyRequest, RDFRequest
from reaxkit.core.analysis_executor import AnalysisExecutor
from reaxkit.core.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.command_alias_resolver import resolve_command_name
from reaxkit.presentation.dispatcher import present_result
from reaxkit.presentation.convert import convert_xaxis

TRAJECTORY_COMMANDS = ("msd", "rdf", "rdf_property")


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None)
    parser.add_argument("--run-dir", "--dir", dest="run_dir", default=".", help="Run directory (fallback for detection)")
    parser.add_argument("--xmolout", "--file", dest="xmolout", default=None, help="Trajectory file path")
    parser.add_argument("--log", choices=["verbose", "quiet"], default=None, help="Logging level")


def _add_presentation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plot", choices=["single", "subplot"], default=None, help="Render a plot")
    parser.add_argument("--show", action="store_true", help="Show the generated plot window")
    parser.add_argument("--save", default=None, help="Save the generated plot to a file path")
    parser.add_argument("--export", default=None, help="Write the result table to CSV")
    parser.add_argument("--grid", default=None, help="Subplot grid like 2x2 or 2*2")
    parser.add_argument("--xaxis", default="frame", choices=["iter", "frame", "time"], help="Quantity on x-axis")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--frames", type=int, nargs="*", default=None, help="Selected frame indices")
    parser.add_argument("--every", type=int, default=1, help="Frame stride")


def _build_msd_request(args: argparse.Namespace) -> MSDRequest:
    return MSDRequest(
        atom_ids=args.atom_ids,
        atom_types=args.atom_types,
        dims=tuple(args.dims),
        origin=args.origin,
        frames=args.frames,
        every=args.every,
    )


def _build_rdf_request(args: argparse.Namespace) -> RDFRequest:
    return RDFRequest(
        atom_ids_a=args.atom_ids_a,
        atom_ids_b=args.atom_ids_b,
        atom_types_a=args.atom_types_a,
        atom_types_b=args.atom_types_b,
        frames=args.frames,
        every=args.every,
        bins=args.bins,
        r_max=args.r_max,
        average=False,
        return_stack=True,
        backend=args.backend,
    )


def _build_rdf_property_request(args: argparse.Namespace) -> RDFPropertyRequest:
    return RDFPropertyRequest(
        property=args.property or args.prop or "first_peak",
        atom_ids_a=args.atom_ids_a,
        atom_ids_b=args.atom_ids_b,
        atom_types_a=args.atom_types_a,
        atom_types_b=args.atom_types_b,
        frames=args.frames,
        every=args.every,
        bins=args.bins,
        r_max=args.r_max,
        backend=args.backend,
    )


REQUEST_BUILDERS: dict[str, Callable[[argparse.Namespace], object]] = {
    "msd": _build_msd_request,
    "rdf": _build_rdf_request,
    "rdf_property": _build_rdf_property_request,
}


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Build the parser for a direct trajectory command."""
    canonical = resolve_command_name(command, task_names=TRAJECTORY_COMMANDS)
    parser.set_defaults(command=canonical)
    parser.formatter_class = argparse.RawTextHelpFormatter

    _add_runtime_arguments(parser)
    _add_presentation_arguments(parser)
    _add_common_arguments(parser)

    if canonical == "msd":
        parser.description = (
            "Compute mean-squared displacement for selected atoms.\n\n"
            "Examples:\n"
            "  reaxkit msd --atom-ids 1 2 3 --plot single\n"
            "  reaxkit msd --atom-types O --xaxis time --save msd_oxygen.png\n"
            "  reaxkit msd --atom-ids 5 --frames 0 10 20 --export msd_atom5.csv"
        )
        parser.add_argument("--atom-ids", type=int, nargs="*", default=None, help="1-based atom ids")
        parser.add_argument("--atom-types", nargs="*", default=None, help="Element symbols to include")
        parser.add_argument("--dims", nargs="*", default=("x", "y", "z"), help="Coordinate dimensions to include")
        parser.add_argument("--origin", default="first", help="Reference frame: 'first' or an explicit index")
    elif canonical == "rdf":
        parser.description = (
            "Compute radial distribution functions for selected atoms.\n\n"
            "Examples:\n"
            "  reaxkit rdf --atom-types-a O --atom-types-b H --plot single\n"
            "  reaxkit rdf --atom-ids-a 1 2 --atom-ids-b 10 11 --bins 300 --save rdf_pairs.png\n"
            "  reaxkit rdf --atom-types-a Al --atom-types-b O --frames 0 50 100 --plot subplot"
        )
        parser.add_argument("--atom-ids-a", type=int, nargs="*", default=None, help="1-based atom ids for group A")
        parser.add_argument("--atom-ids-b", type=int, nargs="*", default=None, help="1-based atom ids for group B")
        parser.add_argument("--atom-types-a", nargs="*", default=None, help="Element symbols for group A")
        parser.add_argument("--atom-types-b", nargs="*", default=None, help="Element symbols for group B")
        parser.add_argument("--bins", type=int, default=200, help="Number of RDF bins")
        parser.add_argument("--r-max", type=float, default=None, help="Maximum radius")
        parser.add_argument("--backend", choices=["freud", "ovito"], default="freud")
    elif canonical == "rdf_property":
        parser.description = (
            "Compute RDF-derived properties across selected frames.\n\n"
            "Examples:\n"
            "  reaxkit rdf_property --property first_peak --atom-types-a O --atom-types-b H --plot single\n"
            "  reaxkit rdf_property --prop area --atom-types-a Al --atom-types-b O --xaxis iter --save rdf_area.png\n"
            "  reaxkit rdf_property --property dominant_peak --frames 0 20 40 --export rdf_peak.csv"
        )
        parser.add_argument("--property", default=None, help="RDF property to extract")
        parser.add_argument("--prop", choices=["first_peak", "dominant_peak", "area", "excess_area"], default=None, help="Legacy alias for --property")
        parser.add_argument("--atom-ids-a", type=int, nargs="*", default=None, help="1-based atom ids for group A")
        parser.add_argument("--atom-ids-b", type=int, nargs="*", default=None, help="1-based atom ids for group B")
        parser.add_argument("--atom-types-a", nargs="*", default=None, help="Element symbols for group A")
        parser.add_argument("--atom-types-b", nargs="*", default=None, help="Element symbols for group B")
        parser.add_argument("--bins", type=int, default=200, help="Number of RDF bins")
        parser.add_argument("--r-max", type=float, default=None, help="Maximum radius")
        parser.add_argument("--backend", choices=["freud", "ovito"], default="freud")
    else:
        raise KeyError(f"Unsupported trajectory command '{canonical}'.")

    return parser


def _plot_payload(command: str, result, args: argparse.Namespace) -> dict[str, object] | None:
    table = result.table
    if table.empty:
        return None

    if command == "msd":
        grouped = table.groupby(["frame_index", "iter", "atom_id"], as_index=False)["msd"].mean()
        frame_values = np.sort(grouped["frame_index"].unique())
        xvals, xlabel = convert_xaxis(frame_values, getattr(args, "xaxis", "frame"))
        frame_to_x = dict(zip(frame_values, xvals))

        series = []
        subplots = []
        for atom_id in sorted(grouped["atom_id"].unique()):
            dfi = grouped[grouped["atom_id"] == atom_id].sort_values("frame_index")
            payload = {
                "x": [frame_to_x[idx] for idx in dfi["frame_index"].to_numpy()],
                "y": dfi["msd"].tolist(),
                "label": f"atom {atom_id}",
            }
            series.append(payload)
            subplots.append([payload])

        title = f"MSD of atoms: {', '.join(str(v) for v in sorted(grouped['atom_id'].unique()))}"
        if getattr(args, "plot", None) == "subplot":
            return {
                "plot_type": "multi_subplots",
                "subplots": subplots,
                "xlabel": xlabel,
                "ylabel": "A^2",
                "title": title,
                "legend": True,
                "grid": getattr(args, "grid", None),
            }
        return {
            "plot_type": "single_plot",
            "series": series,
            "xlabel": xlabel,
            "ylabel": "A^2",
            "title": title,
            "legend": True,
        }

    if command == "rdf":
        if "frame_index" not in table.columns or "iter" not in table.columns:
            return {
                "plot_type": "single_plot",
                "x": table["r"].tolist(),
                "y": table["g"].tolist(),
                "xlabel": "r (A)",
                "ylabel": "g(r)",
                "title": "Radial Distribution Function",
            }

        grouped = table.groupby(["frame_index", "iter"], sort=True)
        series = []
        subplots = []
        for (frame_index, iter_value), dfi in grouped:
            dfi = dfi.sort_values("r")
            payload = {
                "x": dfi["r"].tolist(),
                "y": dfi["g"].tolist(),
                "label": f"frame {frame_index} (iter {iter_value})",
            }
            series.append(payload)
            subplots.append([payload])

        if getattr(args, "plot", None) == "subplot":
            return {
                "plot_type": "multi_subplots",
                "subplots": subplots,
                "xlabel": "r (A)",
                "ylabel": "g(r)",
                "title": "Radial Distribution Function",
                "legend": False,
                "grid": getattr(args, "grid", None),
            }

        return {
            "plot_type": "single_plot",
            "series": series,
            "xlabel": "r (A)",
            "ylabel": "g(r)",
            "title": "Radial Distribution Function",
            "legend": True,
        }

    if command == "rdf_property":
        property_name = getattr(args, "property", None) or getattr(args, "prop", None) or "first_peak"
        property_name = str(property_name).strip().lower()
        x_col = "frame_index"
        xlabel = "Frame Index"
        if getattr(args, "xaxis", "frame") == "iter" and "iter" in table.columns:
            x_col = "iter"
            xlabel = "Iteration"

        property_columns = {
            "first_peak": ("r_first_peak", "First Peak Position"),
            "dominant_peak": ("r_peak", "Dominant Peak Position"),
            "area": ("area", "RDF Area"),
            "excess_area": ("excess_area", "RDF Excess Area"),
        }
        y_col, title = property_columns.get(property_name, (None, None))
        if y_col is None or y_col not in table.columns:
            return None
        return {
            "plot_type": "single_plot",
            "x": table[x_col].tolist(),
            "y": table[y_col].tolist(),
            "xlabel": xlabel,
            "ylabel": y_col,
            "title": title,
        }

    value_columns = [c for c in table.columns if c not in {"frame_index", "iter"}]
    if not value_columns:
        return None
    y_col = value_columns[0]
    x_col = "frame_index"
    if getattr(args, "xaxis", "frame") == "iter" and "iter" in table.columns:
        x_col = "iter"
    return {
        "plot_type": "single_plot",
        "x": table[x_col].tolist(),
        "y": table[y_col].tolist(),
        "xlabel": "Iteration" if x_col == "iter" else "Frame Index",
        "ylabel": y_col,
        "title": f"{command.replace('_', ' ').title()}",
    }


def run_main(command: str, args: argparse.Namespace) -> int:
    """Run a direct trajectory command."""
    canonical = resolve_command_name(command, task_names=TRAJECTORY_COMMANDS)
    task_cls = TASK_REGISTRY[canonical]
    request = REQUEST_BUILDERS[canonical](args)

    executor = AnalysisExecutor()
    result = executor.run(task_cls(), request, vars(args))
    present_result(canonical, result, args, plot_payload_builder=_plot_payload)
    return 0
