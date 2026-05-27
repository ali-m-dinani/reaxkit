"""Direct command workflow for atomic-kinematics analyses."""

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np
import pandas as pd

from reaxkit.analysis import kinematics as _kinematics_tasks  # noqa: F401
from reaxkit.analysis.kinematics.kinematics import AtomicKinematicsRequest
from reaxkit.cli.path import resolve_output_path
from reaxkit.core.analysis_executor import AnalysisExecutor
from reaxkit.core.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.command_alias_resolver import resolve_command_name
from reaxkit.core.storage_layout import add_storage_cli_arguments
from reaxkit.presentation.dispatcher import present_result
from reaxkit.presentation.plot import heatmap2d_from_3d, scatter3d_points

KINEMATICS_COMMANDS = ("kinematics", "kinematics_plot3d", "kinematics_heatmap2d")
KINEMATICS_KEYS = ("metadata", "coordinates", "velocities", "accelerations", "prev_accelerations")


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None, help="Engine override. Example: --engine reaxff, which forces ReaxFF parsing behavior.")
    parser.add_argument("--input", default=".", help="Input file or directory for engine resolution. Example: --input runs/job1, which sets lookup context for required files.")
    parser.add_argument("--run-dir", "--dir", dest="run_dir", default=".", help="Run directory fallback for engine detection. Example: --run-dir runs/job1, which serves as backup path context.")
    parser.add_argument("--vels", "--file", dest="vels", default="vels", help="Atomic kinematics file path. Example: --vels moldyn.vel, which reads kinematics data from that file.")
    parser.add_argument("--log", choices=["verbose", "quiet"], default=None, help="Logging level. Example: --log verbose, which prints more runtime details.")
    add_storage_cli_arguments(parser)


def _add_presentation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plot", choices=["single", "subplot"], default=None, help="Render a plot. Example: --plot subplot, which creates one panel per value column.")
    parser.add_argument("--show", action="store_true", help="Show the generated plot window. Example: --show, which opens the figure interactively.")
    parser.add_argument("--save", default=None, help="Save the generated plot to a file path. Example: --save accelerations.png, which writes the figure image.")
    parser.add_argument("--export", default=None, help="Write the result table to CSV. Example: --export velocities.csv, which saves tabular output.")
    parser.add_argument("--grid", default=None, help="Subplot grid like 2x2 or 2*2. Example: --grid 2x2, which arranges subplot panels in a 2-by-2 layout.")
    parser.add_argument("--xaxis", choices=["atom_index"], default="atom_index", help="Quantity on x-axis. Example: --xaxis atom_index, which uses atom index for horizontal axis.")


def _add_spatial_plot_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--show", action="store_true", help="Show the generated plot window. Example: --show, which opens the rendered spatial plot.")
    parser.add_argument("--save", default=None, help="Save the generated plot to a file path. Example: --save vx_3d.png, which writes the spatial figure to disk.")
    parser.add_argument("--export", default=None, help="Write the merged coordinate/value table to CSV. Example: --export merged.csv, which saves plotted coordinates and values.")
    parser.add_argument("--atoms", type=int, nargs="*", default=None, help="1-based atom ids. Example: --atoms 1 5 9, which limits plotting to those atoms.")
    parser.add_argument("--value", required=True, help="Scalar to plot: vx, vy, vz, ax, ay, az, pax, pay, paz. Example: --value vz, which colors points by z-velocity.")
    parser.add_argument("--vmin", type=float, default=None, help="Color scale minimum. Example: --vmin -0.2, which clamps lower color bound.")
    parser.add_argument("--vmax", type=float, default=None, help="Color scale maximum. Example: --vmax 0.2, which clamps upper color bound.")
    parser.add_argument("--cmap", default="coolwarm", help="Matplotlib colormap. Example: --cmap viridis, which sets the plot color palette.")


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
    parser.set_defaults(progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter

    _add_runtime_arguments(parser)

    if canonical == "kinematics":
        _add_presentation_arguments(parser)
        parser.description = (
            "Extract kinematics datasets from atomic kinematics files.\n"
            "This command can return metadata, coordinates, velocities, accelerations, or previous\n"
            "accelerations, optionally filtered to selected atoms.\n\n"
            "Examples:\n"
            "  1. Export selected-atom velocities:\n"
            "   reaxkit kinematics --key velocities --atoms 1 3 7 --export velocities.csv\n\n"
            "  2. Read metadata from a specific kinematics file:\n"
            "   reaxkit kinematics --key metadata --vels moldyn.vel\n\n"
            "  3. Plot accelerations using subplot layout:\n"
            "   reaxkit kinematics --key accelerations --plot subplot --save accelerations.png"
        )
        parser.add_argument("--key", choices=KINEMATICS_KEYS, required=True, help="Requested kinematics dataset. Example: --key velocities, which returns velocity components by atom.")
        parser.add_argument("--atoms", type=int, nargs="*", default=None, help="1-based atom ids. Example: --atoms 1 3 7, which limits output rows to those atoms.")
    else:
        raise KeyError(f"Unsupported kinematics command '{canonical}'.")

    return parser


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
        out_csv = resolve_output_path(
            args.export,
            command,
            run_id=getattr(args, "run_id", None),
            project_root=getattr(args, "project_root", "."),
            analysis_id=getattr(args, "analysis_id", None),
        )
        merged.to_csv(out_csv, index=False)
        args.export = str(out_csv)

    coords = merged[["x", "y", "z"]].to_numpy(float)
    values = merged[col].to_numpy(float)
    mask = np.isfinite(values)
    coords = coords[mask]
    values = values[mask]
    if coords.size == 0:
        raise ValueError("All selected values are NaN/invalid; nothing to plot.")

    if command == "kinematics_plot3d":
        save_path = None
        if args.save:
            out_save = resolve_output_path(
                args.save,
                command,
                run_id=getattr(args, "run_id", None),
                project_root=getattr(args, "project_root", "."),
                analysis_id=getattr(args, "analysis_id", None),
            )
            save_path = str(out_save)
            args.save = save_path
        scatter3d_points(
            coords,
            values,
            title=f"{args.value}_3D",
            cmap=args.cmap,
            vmin=args.vmin,
            vmax=args.vmax,
            save=save_path,
            show_message=bool(args.show),
        )
        return 0

    save_path = None
    if args.save:
        out_save = resolve_output_path(
            args.save,
            command,
            run_id=getattr(args, "run_id", None),
            project_root=getattr(args, "project_root", "."),
            analysis_id=getattr(args, "analysis_id", None),
        )
        save_path = str(out_save)
        args.save = save_path
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
        save=save_path,
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
    present_result(canonical, result, args, plot_payload_builder=_plot_payload)
    return 0
