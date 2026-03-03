"""Direct command workflow for atomic-kinematics analyses."""

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np
import pandas as pd

from reaxkit.analysis import kinematics as _kinematics_tasks  # noqa: F401
from reaxkit.analysis.kinematics.kinematics import AtomicKinematicsRequest
from reaxkit.core.analysis_executor import AnalysisExecutor
from reaxkit.core.task_registry import TASK_REGISTRY
from reaxkit.core.task_resolution_using_alias import resolve_command_name
from reaxkit.presentation.dispatcher import present_result
from reaxkit.presentation.plot import heatmap2d_from_3d, scatter3d_points

KINEMATICS_COMMANDS = ("kinematics", "kinematics_plot3d", "kinematics_heatmap2d")
KINEMATICS_KEYS = ("metadata", "coordinates", "velocities", "accelerations", "prev_accelerations")


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None)
    parser.add_argument("--input", default=".", help="Input file or directory for engine resolution")
    parser.add_argument("--run-dir", "--dir", dest="run_dir", default=".", help="Run directory fallback for engine detection")
    parser.add_argument("--vels", "--file", dest="vels", default="vels", help="Atomic kinematics file path")
    parser.add_argument("--log", choices=["verbose", "quiet"], default=None, help="Logging level")


def _add_presentation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plot", choices=["single", "subplot"], default=None, help="Render a plot")
    parser.add_argument("--show", action="store_true", help="Show the generated plot window")
    parser.add_argument("--save", default=None, help="Save the generated plot to a file path")
    parser.add_argument("--export", default=None, help="Write the result table to CSV")
    parser.add_argument("--grid", default=None, help="Subplot grid like 2x2 or 2*2")
    parser.add_argument("--xaxis", choices=["atom_index"], default="atom_index", help="Quantity on x-axis")


def _add_spatial_plot_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--show", action="store_true", help="Show the generated plot window")
    parser.add_argument("--save", default=None, help="Save the generated plot to a file path")
    parser.add_argument("--export", default=None, help="Write the merged coordinate/value table to CSV")
    parser.add_argument("--atoms", type=int, nargs="*", default=None, help="1-based atom ids")
    parser.add_argument("--value", required=True, help="Scalar to plot: vx, vy, vz, ax, ay, az, pax, pay, paz")
    parser.add_argument("--vmin", type=float, default=None, help="Color scale minimum")
    parser.add_argument("--vmax", type=float, default=None, help="Color scale maximum")
    parser.add_argument("--cmap", default="coolwarm", help="Matplotlib colormap")


def _parse_bins(bins: str):
    if "," in bins:
        nx, ny = [int(x) for x in bins.split(",")]
        return (nx, ny)
    return int(bins)


def _value_column_to_key(value_col: str) -> tuple[str, str]:
    value = value_col.strip().lower()
    if value in {"vx", "vy", "vz"}:
        return ("velocities", value)
    if value in {"ax", "ay", "az"}:
        return ("accelerations", value)
    if value in {"pax", "pay", "paz"}:
        return ("prev_accelerations", "a" + value[1:])
    raise ValueError("value must be one of: vx, vy, vz, ax, ay, az, pax, pay, paz")


def _build_atomic_kinematics_request(args: argparse.Namespace) -> AtomicKinematicsRequest:
    return AtomicKinematicsRequest(
        key=args.key,
        atoms=args.atoms,
    )


REQUEST_BUILDERS: dict[str, Callable[[argparse.Namespace], object]] = {
    "kinematics": _build_atomic_kinematics_request,
}


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    canonical = resolve_command_name(command, task_names=KINEMATICS_COMMANDS)
    parser.set_defaults(command=canonical)
    parser.formatter_class = argparse.RawTextHelpFormatter

    _add_runtime_arguments(parser)

    if canonical == "kinematics":
        _add_presentation_arguments(parser)
        parser.description = (
            "Extract metadata or atomic coordinate, velocity, or acceleration tables.\n\n"
            "Examples:\n"
            "  reaxkit kinematics --key velocities --atoms 1 3 7 --export velocities.csv\n"
            "  reaxkit kinematics --key metadata --vels moldyn.vel\n"
            "  reaxkit kinematics --key accelerations --plot subplot --save accelerations.png"
        )
        parser.add_argument("--key", choices=KINEMATICS_KEYS, required=True, help="Requested kinematics dataset")
        parser.add_argument("--atoms", type=int, nargs="*", default=None, help="1-based atom ids")
    elif canonical == "kinematics_plot3d":
        _add_spatial_plot_arguments(parser)
        parser.description = (
            "Plot atomic velocity or acceleration values on 3D atomic coordinates.\n\n"
            "Examples:\n"
            "  reaxkit kinematics_plot3d --value vz --save vz_3d.png\n"
            "  reaxkit kinematics_plot3d --vels moldyn.vel --value ax --atoms 1 3 7 --show\n"
            "  reaxkit kinematics_plot3d --vels molsav.0001 --value paz --export paz_coords.csv"
        )
    elif canonical == "kinematics_heatmap2d":
        _add_spatial_plot_arguments(parser)
        parser.description = (
            "Project atomic velocity or acceleration values into a 2D heatmap.\n\n"
            "Examples:\n"
            "  reaxkit kinematics_heatmap2d --value vz --plane xz --bins 60 --save vz_xz.png\n"
            "  reaxkit kinematics_heatmap2d --vels moldyn.vel --value ax --plane xy --agg max --show\n"
            "  reaxkit kinematics_heatmap2d --value paz --plane yz --export paz_yz.csv"
        )
        parser.add_argument("--plane", default="xy", choices=["xy", "xz", "yz"], help="Projection plane")
        parser.add_argument("--bins", default="50", help='Grid bins: "N" or "Nx,Ny"')
        parser.add_argument("--agg", default="mean", help="Aggregation: mean|max|min|sum|count")
    else:
        raise KeyError(f"Unsupported kinematics command '{canonical}'.")

    return parser


def _metadata_to_table(result) -> None:
    if result.table is not None or not result.metadata:
        return
    rows = [{"key": key, "value": value} for key, value in result.metadata.items()]
    result.table = pd.DataFrame(rows)


def _plot_payload(command: str, result, args: argparse.Namespace) -> dict[str, object] | None:
    table = result.table
    if not isinstance(table, pd.DataFrame) or table.empty:
        return None
    if command != "kinematics" or getattr(args, "key", None) == "metadata":
        return None
    if "atom_index" not in table.columns:
        return None

    value_columns = [col for col in table.columns if col != "atom_index"]
    if not value_columns:
        return None

    if getattr(args, "plot", None) == "subplot":
        subplots = []
        for col in value_columns:
            subplots.append([{"x": table["atom_index"].tolist(), "y": pd.to_numeric(table[col], errors="coerce").tolist(), "label": str(col)}])
        return {
            "plot_type": "multi_subplots",
            "subplots": subplots,
            "xlabel": "Atom Index",
            "ylabel": "Value",
            "title": str(args.key).replace("_", " ").title(),
            "legend": False,
            "grid": getattr(args, "grid", None),
        }

    series = []
    for col in value_columns:
        series.append({"x": table["atom_index"].tolist(), "y": pd.to_numeric(table[col], errors="coerce").tolist(), "label": str(col)})
    return {
        "plot_type": "single_plot",
        "series": series,
        "xlabel": "Atom Index",
        "ylabel": "Value",
        "title": str(args.key).replace("_", " ").title(),
        "legend": True,
    }


def _run_spatial(command: str, args: argparse.Namespace) -> int:
    executor = AnalysisExecutor()
    task_cls = TASK_REGISTRY["atomic_kinematics"]

    coords_result = executor.run(task_cls(), AtomicKinematicsRequest(key="coordinates", atoms=args.atoms), vars(args))
    key, col = _value_column_to_key(args.value)
    values_result = executor.run(task_cls(), AtomicKinematicsRequest(key=key, atoms=args.atoms), vars(args))

    cdf = coords_result.table
    vdf = values_result.table
    if not isinstance(cdf, pd.DataFrame) or cdf.empty:
        raise ValueError("No coordinates available for plotting.")
    if not isinstance(vdf, pd.DataFrame) or vdf.empty:
        raise ValueError(f"No data available for {args.value}.")

    merged = cdf.merge(vdf, on="atom_index", how="inner")
    if merged.empty:
        raise ValueError("No matching atom rows between coordinates and selected values.")

    if args.export:
        merged.to_csv(args.export, index=False)

    coords = merged[["x", "y", "z"]].to_numpy(float)
    values = merged[col].to_numpy(float)
    mask = np.isfinite(values)
    coords = coords[mask]
    values = values[mask]
    if coords.size == 0:
        raise ValueError("All selected values are NaN/invalid; nothing to plot.")

    if command == "kinematics_plot3d":
        scatter3d_points(
            coords,
            values,
            title=f"{args.value}_3D",
            cmap=args.cmap,
            vmin=args.vmin,
            vmax=args.vmax,
            save=args.save,
            show_message=bool(args.show),
        )
        return 0

    heatmap2d_from_3d(
        coords,
        values,
        plane=args.plane,
        bins=_parse_bins(args.bins),
        agg=args.agg,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap,
        title=f"{args.value}_{args.plane}_heatmap2d",
        save=args.save,
        show_message=bool(args.show),
    )
    return 0


def run_main(command: str, args: argparse.Namespace) -> int:
    canonical = resolve_command_name(command, task_names=KINEMATICS_COMMANDS)
    if canonical in {"kinematics_plot3d", "kinematics_heatmap2d"}:
        return _run_spatial(canonical, args)

    task_cls = TASK_REGISTRY["atomic_kinematics"]
    request = REQUEST_BUILDERS[canonical](args)

    executor = AnalysisExecutor()
    result = executor.run(task_cls(), request, vars(args))
    _metadata_to_table(result)
    present_result(canonical, result, args, plot_payload_builder=_plot_payload)
    return 0
