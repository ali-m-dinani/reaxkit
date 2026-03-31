"""Command workflow for electrostatics analyses and local visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Union

import numpy as np
import pandas as pd
import reaxkit.engine  # noqa: F401

from reaxkit.analysis import electrostatics as _electrostatics_tasks  # noqa: F401
from reaxkit.analysis.electrostatics.charges import ChargeTableRequest
from reaxkit.analysis.electrostatics.electrostatics import (
    DipoleRequest,
    PolarizationFieldRequest,
    PolarizationRequest,
)
from reaxkit.cli.path import resolve_output_path
from reaxkit.core.analysis_executor import AnalysisExecutor
from reaxkit.core.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.alias import _resolve_alias, normalize_choice, resolve_alias_from_columns
from reaxkit.core.engine_registry import resolve_engine
from reaxkit.core.frame_utils import parse_frame_indices, parse_frames
from reaxkit.core.command_alias_resolver import resolve_command_name
from reaxkit.core.progress import resolve_reporter
from reaxkit.core.storage_layout import add_storage_cli_arguments
from reaxkit.domain.data_models import (
    TrajectoryData,
)
from reaxkit.presentation.convert import convert_xaxis
from reaxkit.presentation.dispatcher import export_result_csv, present_result
from reaxkit.presentation.plot import plot as render_plot

ELECTROSTATICS_COMMANDS = ("charge_table", "dipole", "polarization", "polarization_field")


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None)
    parser.add_argument("--run-dir", "--dir", dest="run_dir", default=".", help="Run directory fallback for detection")
    parser.add_argument("--xmolout", default="xmolout", help="Path to xmolout file")
    parser.add_argument("--fort7", default="fort.7", help="Path to fort.7 file")
    add_storage_cli_arguments(parser)


def _add_scalar_presentation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--export", default=None, help="Write the analysis table to CSV")
    parser.add_argument(
        "--plot",
        choices=["plot3d", "heatmap2d"],
        default=None,
        help="Render local dipole or polarization results as a plot",
    )
    parser.add_argument("--component", default=None, help="Component to color by for local plots")
    parser.add_argument(
        "--plot-frames",
        nargs="*",
        default=None,
        help='Frames for local plots: "0,10,20", "0 10 20", "0:20", "0-20", or "0:20:2" (defaults to --frames selection)',
    )
    parser.add_argument("--save", default=None, help="Directory used when saving local plots")
    parser.add_argument("--vmin", type=float, default=None, help="Color scale minimum for local plots")
    parser.add_argument("--vmax", type=float, default=None, help="Color scale maximum for local plots")
    parser.add_argument("--cmap", default=None, help="Matplotlib colormap for local plots")
    parser.add_argument("--plane", default="xy", choices=["xy", "xz", "yz"], help="Projection plane for heatmaps")
    parser.add_argument("--bins", default="40", help='Grid bins for heatmaps: "N" or "Nx,Ny"')
    parser.add_argument("--agg", default="mean", help="Heatmap aggregation: mean|max|min|sum|count")
    parser.add_argument("--size", type=float, default=20.0, help="3D marker size")
    parser.add_argument("--alpha", type=float, default=0.9, help="3D marker transparency")
    parser.add_argument("--elev", type=float, default=22.0, help="3D view elevation")
    parser.add_argument("--azim", type=float, default=38.0, help="3D view azimuth")


def _add_table_presentation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plot", choices=["single", "subplot"], default=None, help="Render a plot")
    parser.add_argument("--show", action="store_true", help="Show the generated plot window")
    parser.add_argument("--save", default=None, help="Save the generated plot to a file path")
    parser.add_argument("--export", default=None, help="Write the table to CSV")
    parser.add_argument("--grid", default=None, help="Subplot grid like 2x2 or 2*2")
    parser.add_argument("--xaxis", choices=["iter", "frame", "time"], default="iter", help="Quantity on x-axis")


def _parse_core_types(spec: str | None) -> tuple[str, ...]:
    if not spec:
        return ()
    return tuple(token.strip() for token in spec.split(",") if token.strip())


def _parse_bins(spec: str) -> Union[int, tuple[int, int]]:
    if "," in spec:
        nx, ny = [int(token.strip()) for token in spec.split(",", maxsplit=1)]
        return (nx, ny)
    return int(spec)


def _parse_frame_indices(n_frames: int, spec) -> list[int]:
    if not spec:
        return list(range(n_frames))

    selector = parse_frames(spec)
    if selector is None:
        return list(range(n_frames))
    if isinstance(selector, slice):
        start = 0 if selector.start is None else int(selector.start)
        stop = n_frames if selector.stop is None else int(selector.stop)
        step = 1 if selector.step is None else int(selector.step)
        return [idx for idx in range(start, min(stop, n_frames), step) if 0 <= idx < n_frames]
    return [int(idx) for idx in selector if 0 <= int(idx) < n_frames]


def _parse_frame_selector(spec) -> list[int] | None:
    return parse_frame_indices(spec)


def _detection_path(args_map: dict) -> str:
    source_dir = args_map.get("_snapshot_source_dir")
    if source_dir:
        p = Path(str(source_dir))
        if p.exists():
            return str(p)
    for key in ("xmolout", "fort7", "fort78", "control", "run_dir"):
        value = args_map.get(key)
        if value:
            return str(value)
    return "."


def _resolve_adapter(args: argparse.Namespace):
    args_map = vars(args)
    return resolve_engine(_detection_path(args_map), engine=getattr(args, "engine", None))


def _build_dipole_request(args: argparse.Namespace) -> DipoleRequest:
    scope = str(args.scope)
    core_types = _parse_core_types(args.core) if scope == "local" else ()
    if scope == "local" and not core_types:
        raise ValueError("When --scope local is used, --core must be provided (for example --core Al,Mg).")
    frames = _parse_frame_selector(getattr(args, "frames", None))
    return DipoleRequest(
        scope=scope,
        atom_types=core_types if scope == "local" else None,
        frames=frames,
    )


def _build_polarization_request(args: argparse.Namespace) -> PolarizationRequest:
    scope = str(args.scope)
    core_types = _parse_core_types(args.core) if scope == "local" else ()
    if scope == "local" and not core_types:
        raise ValueError("When --scope local is used, --core must be provided (for example --core Al,Mg).")
    frames = _parse_frame_selector(getattr(args, "frames", None))
    return PolarizationRequest(
        scope=scope,
        atom_types=core_types if scope == "local" else None,
        frames=frames,
        volume_method="bbox" if scope == "local" else "hull",
    )


def _build_polarization_field_request(args: argparse.Namespace) -> PolarizationFieldRequest:
    return PolarizationFieldRequest(
        field_direction="z",
        aggregate=args.aggregate,
        dipole_or_polaization_direction="p_z",
    )


def _build_charge_table_request(args: argparse.Namespace) -> ChargeTableRequest:
    atom_ids = [int(atom_id) for atom_id in args.atom_ids] if args.atom_ids else None
    atom_types = list(args.atom_types) if args.atom_types else None
    return ChargeTableRequest(
        atom_ids=atom_ids,
        atom_types=atom_types,
        frames=_parse_frame_selector(args.frames),
        every=int(args.every),
    )


REQUEST_BUILDERS: dict[str, Callable[[argparse.Namespace], object]] = {
    "charge_table": _build_charge_table_request,
    "dipole": _build_dipole_request,
    "polarization": _build_polarization_request,
    "polarization_field": _build_polarization_field_request,
}


def _summary_text(result) -> str:
    def _fmt(values) -> str:
        if not values:
            return "None found"
        return ", ".join(f"{value:.6g}" for value in values)

    field_dir = getattr(getattr(result, "request", None), "field_direction", "z")
    resp_dir = getattr(getattr(result, "request", None), "dipole_or_polaization_direction", "p_z")
    field_col = f"field_{field_dir}"
    y_map = {
        "mu_x": "mu_x (debye)",
        "mu_y": "mu_y (debye)",
        "mu_z": "mu_z (debye)",
        "p_x": "P_x (uC/cm^2)",
        "p_y": "P_y (uC/cm^2)",
        "p_z": "P_z (uC/cm^2)",
    }
    resp_col = y_map.get(str(resp_dir), "P_z (uC/cm^2)")

    return (
        "Hysteresis Analysis Summary\n"
        "===========================\n\n"
        f"Coercive fields (where {resp_col} crosses zero vs {field_col})\n"
        "Units: MV/cm\n"
        f"Values: {_fmt(result.polarization_zero_crossings)}\n\n"
        f"Remnant responses (where {field_col} crosses zero vs {resp_col})\n"
        "Units: response-axis units\n"
        f"Values: {_fmt(result.field_zero_crossings)}\n"
    )


def _plot_payload(command: str, result, args: argparse.Namespace) -> dict[str, object] | None:
    if command == "charge_table":
        table = result.table
        if not isinstance(table, pd.DataFrame) or table.empty:
            return None

        work = table.copy()
        if args.xaxis == "time" and "time" not in work.columns and "iter" in work.columns:
            converted, xlabel = convert_xaxis(work["iter"].to_numpy(dtype=int), "time", control_file=getattr(args, "control", "control"))
            work["time"] = np.asarray(converted, dtype=float)
            x_col = "time"
        elif args.xaxis == "frame":
            x_col = "frame_index"
            xlabel = "Frame Index"
        else:
            x_col = "iter"
            xlabel = "Time" if args.xaxis == "time" and "time" in work.columns else ("Iteration" if x_col == "iter" else "Frame Index")
            if args.xaxis == "time" and "time" in work.columns:
                x_col = "time"

        if getattr(args, "plot", None) == "subplot":
            subplots = []
            for atom_id, group in work.groupby("atom_id", sort=True):
                subplots.append(
                    [{
                        "x": group[x_col].tolist(),
                        "y": pd.to_numeric(group["charge"], errors="coerce").tolist(),
                        "label": f"atom {atom_id}",
                    }]
                )
            if not subplots:
                return None
            return {
                "plot_type": "multi_subplots",
                "subplots": subplots,
                "xlabel": xlabel,
                "ylabel": "Charge",
                "title": "Charge Table",
                "legend": False,
                "grid": getattr(args, "grid", None),
            }

        series = []
        for atom_id, group in work.groupby("atom_id", sort=True):
            series.append(
                {
                    "x": group[x_col].tolist(),
                    "y": pd.to_numeric(group["charge"], errors="coerce").tolist(),
                    "label": f"atom {atom_id}",
                }
            )
        if not series:
            return None
        return {
            "plot_type": "single_plot",
            "series": series,
            "xlabel": xlabel,
            "ylabel": "Charge",
            "title": "Charge Table",
            "legend": len(series) > 1,
        }

    if command != "polarization_field":
        return None

    canonical_x = normalize_choice(args.xaxis or "field_z", domain="xaxis")
    canonical_y = normalize_choice(args.yaxis or "P_z (uC/cm^2)", domain="yaxis")
    table = result.full_table if canonical_x == "time" else result.aggregated_table
    if table.empty:
        return None

    x_col = _resolve_alias(table, canonical_x)
    y_col = _resolve_alias(table, canonical_y)
    return {
        "plot_type": "single_plot",
        "x": table[x_col].tolist(),
        "y": table[y_col].tolist(),
        "xlabel": x_col,
        "ylabel": y_col,
        "title": f"{y_col} vs {x_col}",
    }


def _default_component_for(command: str) -> str:
    return "P_z (uC/cm^2)" if command == "polarization" else "mu_z (debye)"


def _local_result_with_coords(
    trajectory: TrajectoryData,
    result_table: pd.DataFrame,
    frames_spec: str | None,
) -> pd.DataFrame:
    if result_table.empty:
        raise ValueError("No local electrostatics data found for the requested analysis.")

    n_frames = int(trajectory.positions.shape[0])
    frame_indices = _parse_frame_indices(n_frames, frames_spec)
    df_local = result_table[result_table["frame_index"].isin(frame_indices)].copy()
    if df_local.empty:
        raise ValueError("No local electrostatics data found in the requested frames.")

    coords = np.asarray(trajectory.positions, dtype=float)
    atom_ids = [int(atom_id) for atom_id in trajectory.atom_ids]
    coord_rows: list[dict[str, float | int]] = []
    for frame_index in frame_indices:
        if not (0 <= frame_index < coords.shape[0]):
            continue
        frame_coords = coords[frame_index]
        for atom_id, xyz in zip(atom_ids, frame_coords, strict=False):
            coord_rows.append(
                {
                    "frame_index": int(frame_index),
                    "core_atom_id": int(atom_id),
                    "x": float(xyz[0]),
                    "y": float(xyz[1]),
                    "z": float(xyz[2]),
                }
            )

    coord_df = pd.DataFrame(coord_rows)
    merged = pd.merge(df_local, coord_df, on=["frame_index", "core_atom_id"], how="inner")
    if merged.empty:
        raise ValueError("Could not match local electrostatics entries to core atom coordinates.")
    return merged


def _iter_local_plot_payloads(command: str, table: pd.DataFrame, args: argparse.Namespace) -> list[dict[str, object]]:
    component = args.component or _default_component_for(command)
    col = resolve_alias_from_columns(table.columns, component)
    if col is None:
        raise ValueError(f"Component '{component}' not found. Available columns include: {list(table.columns)[:12]} ...")

    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    payloads: list[dict[str, object]] = []
    for frame_index in sorted(table["frame_index"].unique()):
        sub = table[table["frame_index"] == frame_index]
        coords = sub[["x", "y", "z"]].to_numpy(float)
        values = sub[col].to_numpy(float)
        if coords.size == 0 or not np.isfinite(values).any():
            continue

        mask = np.isfinite(values)
        coords = coords[mask]
        values = values[mask]
        if coords.size == 0:
            continue

        vmin = args.vmin if args.vmin is not None else float(np.nanmin(values))
        vmax = args.vmax if args.vmax is not None else float(np.nanmax(values))

        if args.plot == "plot3d":
            title = f"{col}_local_3D_frame_{frame_index}"
            payloads.append(
                {
                    "plot_type": "scatter3d_points",
                    "coords": coords,
                    "values": values,
                    "title": title,
                    "s": args.size,
                    "alpha": args.alpha,
                    "cmap": args.cmap or "coolwarm",
                    "vmin": vmin,
                    "vmax": vmax,
                    "elev": args.elev,
                    "azim": args.azim,
                    "save": (save_dir / f"{title}.png") if save_dir else None,
                    "show_message": False,
                }
            )
            continue

        title = f"{col}_local_{args.plane}_frame_{frame_index}"
        payloads.append(
            {
                "plot_type": "heatmap2d_from_3d",
                "coords": coords,
                "values": values,
                "plane": args.plane,
                "bins": _parse_bins(args.bins),
                "agg": args.agg,
                "vmin": vmin,
                "vmax": vmax,
                "cmap": args.cmap or "viridis",
                "title": title,
                "save": (save_dir / f"{title}.png") if save_dir else None,
                "show_message": False,
            }
        )
    return payloads


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    canonical = resolve_command_name(command, task_names=ELECTROSTATICS_COMMANDS)
    parser.set_defaults(command=canonical)
    parser.set_defaults(progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter
    _add_runtime_arguments(parser)

    if canonical in {"dipole", "polarization"}:
        parser.description = (
            f"Compute {canonical} data for one or more frames.\n\n"
            "Examples:\n"
            f"  reaxkit {canonical} --frames 10 --scope total --export {canonical}_frame10.csv\n"
            f"  reaxkit {canonical} --frames 0:20:2 --scope total --export {canonical}_series.csv\n"
            f"  reaxkit {canonical} --frames 10 --scope local --core Al --export local_{canonical}_frame10.csv\n"
            f"  reaxkit {canonical} --frames 10 --scope local --core Al --plot plot3d --component z --save {canonical}_plots"
        )
        parser.add_argument(
            "--frames",
            nargs="*",
            default=None,
            help='Frames to analyze: "0,10,20", "0 10 20", "0:20", "0-20", or "0:20:2"',
        )
        parser.add_argument("--scope", choices=["total", "local"], default="total", help="Electrostatics scope")
        parser.add_argument("--core", default=None, help="Comma-separated core atom types for local scope")
        _add_scalar_presentation_arguments(parser)
    elif canonical == "charge_table":
        parser.description = (
            "Extract per-atom charges across selected frames.\n\n"
            "Examples:\n"
            "  reaxkit charge_table --frames 0:10:2 --export charges.csv\n"
            "  reaxkit charge_table --atom-ids 1 2 3 --frames 0,5,10 --plot single\n"
            "  reaxkit charge_table --atom-types O H --every 5 --xaxis time --save charge_series.png"
        )
        parser.add_argument("--atom-ids", type=int, nargs="*", default=None, help="1-based atom ids to include")
        parser.add_argument("--atom-types", nargs="*", default=None, help="Element symbols to include")
        parser.add_argument(
            "--frames",
            nargs="*",
            default=None,
            help='Frames: "0,10,20", "0 10 20", "0:20", "0-20", or "0:20:2"',
        )
        parser.add_argument("--every", type=int, default=1, help="Use every Nth selected frame")
        parser.add_argument("--control", default="control", help="Control file for time-axis conversion")
        _add_table_presentation_arguments(parser)
    elif canonical == "polarization_field":
        parser.description = (
            "Analyze polarization-field hysteresis behavior.\n\n"
            "Examples:\n"
            "  reaxkit polarization_field --plot --save hysteresis.png\n"
            "  reaxkit polarization_field --xaxis field_z --yaxis pol_z --aggregate mean --export hysteresis.csv\n"
            "  reaxkit polarization_field --roots --summary hysteresis_summary.txt"
        )
        parser.add_argument("--fort78", default="fort.78", help="Path to fort.78 file")
        parser.add_argument("--control", default="control", help="Path to control file")
        parser.add_argument("--plot", action="store_true", help="Render the hysteresis plot")
        parser.add_argument("--save", default="hysteresis_aggregated.png", help="Save plot to a file path")
        parser.add_argument("--export", default="hysteresis_aggregated.csv", help="Write aggregated data to CSV")
        parser.add_argument(
            "--summary",
            default="hysteresis_summary.txt",
            help="Write coercive fields and remnant polarizations to a text file",
        )
        parser.add_argument("--yaxis", default="pol_z", help="Quantity for y-axis")
        parser.add_argument("--xaxis", default="field_z", help="Quantity for x-axis")
        parser.add_argument(
            "--aggregate",
            choices=["mean", "max", "min", "last"],
            default="mean",
            help="Aggregation method",
        )
        parser.add_argument("--roots", action="store_true", help="Also print coercive and remnant values to stdout")
    else:
        raise KeyError(f"Unsupported electrostatics command '{canonical}'.")

    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    """Run a direct electrostatics command."""
    canonical = resolve_command_name(command, task_names=ELECTROSTATICS_COMMANDS)
    if canonical in {"dipole", "polarization"} and not getattr(args, "frames", None):
        raise ValueError("Provide --frames selector (for example --frames 0:20:2).")

    task_cls = TASK_REGISTRY[canonical]
    request = REQUEST_BUILDERS[canonical](args)

    executor = AnalysisExecutor()
    result = executor.run(task_cls(), request, vars(args))

    if canonical == "charge_table" and isinstance(getattr(result, "table", None), pd.DataFrame):
        if "time" not in result.table.columns and "iter" in result.table.columns:
            try:
                converted, _ = convert_xaxis(
                    result.table["iter"].to_numpy(dtype=int),
                    "time",
                    control_file=getattr(args, "control", "control"),
                )
            except Exception:
                converted = None
            if converted is not None:
                result.table = result.table.copy()
                result.table["time"] = np.asarray(converted, dtype=float)

    if canonical == "polarization_field":
        result.table = result.aggregated_table
        if "time" not in result.full_table.columns and "iter" in result.full_table.columns:
            converted, _ = convert_xaxis(
                result.full_table["iter"].to_numpy(dtype=int),
                "time",
                control_file=getattr(args, "control", "control"),
            )
            result.full_table = result.full_table.copy()
            result.full_table["time"] = np.asarray(converted, dtype=float)

    local_plot_requested = canonical in {"dipole", "polarization"} and getattr(args, "plot", None) in {"plot3d", "heatmap2d"}
    args_for_present = args
    if local_plot_requested:
        args_for_present = argparse.Namespace(**vars(args))
        args_for_present.plot = None
        args_for_present.save = None

    present_result(canonical, result, args_for_present, plot_payload_builder=_plot_payload)

    if local_plot_requested:
        if request.scope != "local":
            raise ValueError(f"--plot {args.plot} requires --scope local.")
        reporter = resolve_reporter(vars(args))
        adapter = _resolve_adapter(args)
        trajectory = adapter.load(TrajectoryData, vars(args), reporter=reporter)
        plot_frames_spec = args.plot_frames if getattr(args, "plot_frames", None) else args.frames
        local_table = _local_result_with_coords(trajectory, result.table, plot_frames_spec)
        payloads = _iter_local_plot_payloads(canonical, local_table, args)
        for payload in payloads:
            render_plot(payload)

    if canonical == "polarization_field":
        if getattr(args, "export", None):
            out_csv = resolve_output_path(
                args.export,
                canonical,
                run_id=getattr(args, "run_id", None),
                project_root=getattr(args, "project_root", "."),
                analysis_id=getattr(args, "analysis_id", None),
            )
            full_save_path = out_csv.parent / "hysteresis_full_data.csv"
            export_result_csv(SimpleNamespace(table=result.full_table), str(full_save_path))
        if getattr(args, "summary", None):
            summary_path = Path(args.summary)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(_summary_text(result), encoding="utf-8")
        if getattr(args, "roots", False):
            coercive = ", ".join(f"{value:.6g}" for value in result.polarization_zero_crossings) or "None found"
            remnant = ", ".join(f"{value:.6g}" for value in result.field_zero_crossings) or "None found"
            print("\n[Hysteresis roots]")
            print("  Coercive fields (MV/cm):", coercive)
            print("  Remnant polarizations (uC/cm^2):", remnant)

    return 0
