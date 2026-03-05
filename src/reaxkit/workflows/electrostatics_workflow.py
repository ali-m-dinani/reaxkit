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
from reaxkit.analysis.electrostatics.charges import ChargeTableRequest, ChargeTableTask
from reaxkit.analysis.electrostatics.electrostatics import (
    DipoleRequest,
    DipoleTask,
    PolarizationFieldRequest,
    PolarizationFieldTask,
    PolarizationRequest,
    PolarizationTask,
)
from reaxkit.cli.path import resolve_output_path
from reaxkit.core.alias import _resolve_alias, normalize_choice, resolve_alias_from_columns
from reaxkit.core.engine_registry import resolve_engine
from reaxkit.core.frame_utils import parse_frames
from reaxkit.core.command_alias_resolver import resolve_command_name
from reaxkit.core.storage_layout import add_storage_cli_arguments, normalize_storage_args
from reaxkit.domain.data_models import (
    ChargeData,
    ConnectivityData,
    ElectricFieldData,
    ElectrostaticsData,
    TrajectoryData,
)
from reaxkit.presentation.convert import convert_xaxis
from reaxkit.presentation.dispatcher import export_result_csv, present_result, print_result_table
from reaxkit.presentation.plot import plot as render_plot

ELECTROSTATICS_COMMANDS = ("charge_table", "dipole", "polarization", "hyst")


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
    parser.add_argument("--frames", default=None, help='Frames for local plots: "0,10,20" or "0:100:5"')
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


def _parse_frame_indices(n_frames: int, spec: str | None) -> list[int]:
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


def _parse_frame_selector(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    selector = parse_frames(spec)
    if selector is None:
        return None
    if isinstance(selector, slice):
        start = 0 if selector.start is None else int(selector.start)
        stop = int(selector.stop) if selector.stop is not None else start
        step = 1 if selector.step is None else int(selector.step)
        return list(range(start, stop, step))
    return [int(idx) for idx in selector]


def _normalized_args(args: argparse.Namespace) -> dict:
    normalized = normalize_storage_args(vars(args))
    for key, value in normalized.items():
        setattr(args, key, value)
    return normalized


def _detection_path(args_map: dict) -> str:
    for key in ("xmolout", "fort7", "fort78", "control", "run_dir"):
        value = args_map.get(key)
        if value:
            return str(value)
    return "."


def _resolve_adapter(args: argparse.Namespace):
    args_map = _normalized_args(args)
    return resolve_engine(_detection_path(args_map), engine=getattr(args, "engine", None))


def _load_electrostatics_data(args: argparse.Namespace) -> ElectrostaticsData:
    adapter = _resolve_adapter(args)
    load_args = _normalized_args(args)
    trajectory = adapter.load(TrajectoryData, load_args)
    charges = adapter.load(ChargeData, load_args)
    connectivity = adapter.load(ConnectivityData, load_args)
    return ElectrostaticsData(
        trajectory=trajectory,
        charges=charges,
        connectivity=connectivity,
    )


def _extract_field_component(field_data: ElectricFieldData, component: str) -> np.ndarray:
    names = [str(name) for name in field_data.applied_field_components]
    values = np.asarray(field_data.applied_field_values, dtype=float)
    if component not in names:
        raise KeyError(f"Electric field component '{component}' not found in {names}.")
    if values.ndim == 1:
        if len(names) != 1:
            raise ValueError("1D electric field values require exactly one component label.")
        return values
    col = names.index(component)
    return values[:, col]


def _align_field_to_iterations(
    field_data: ElectricFieldData,
    *,
    target_iters: np.ndarray,
    component: str,
) -> ElectricFieldData:
    raw_values = _extract_field_component(field_data, component)
    if field_data.sampled_field_iterations is None:
        if len(raw_values) < len(target_iters):
            raise ValueError("Electric field series shorter than selected polarization frames.")
        aligned = np.asarray(raw_values[: len(target_iters)], dtype=float)
    else:
        field_iters = np.asarray(field_data.sampled_field_iterations, dtype=int).reshape(-1)
        if field_iters.shape[0] != raw_values.shape[0]:
            raise ValueError("Electric field sample count does not match sampled_field_iterations.")
        order = np.argsort(field_iters)
        sorted_iters = field_iters[order]
        sorted_values = np.asarray(raw_values, dtype=float)[order]
        aligned = np.empty((len(target_iters),), dtype=float)
        for idx, iter_value in enumerate(np.asarray(target_iters, dtype=int)):
            valid = np.where(sorted_iters <= int(iter_value))[0]
            if valid.size == 0:
                aligned[idx] = 0.0 if int(iter_value) == 0 else np.nan
            else:
                aligned[idx] = float(sorted_values[valid[-1]])

    return ElectricFieldData(
        applied_field_values=aligned,
        applied_field_components=(component,),
        sampled_field_iterations=np.asarray(target_iters, dtype=int),
    )


def _load_hysteresis_data(args: argparse.Namespace) -> ElectrostaticsData:
    adapter = _resolve_adapter(args)
    load_args = _normalized_args(args)
    data = _load_electrostatics_data(args)
    raw_field = adapter.load(ElectricFieldData, load_args)
    target_iters = np.asarray(
        data.trajectory.iterations
        if data.trajectory.iterations is not None
        else np.arange(data.trajectory.positions.shape[0]),
        dtype=int,
    )
    data.electric_field = _align_field_to_iterations(raw_field, target_iters=target_iters, component="field_z")
    return data


def _build_dipole_request(args: argparse.Namespace) -> DipoleRequest:
    scope = str(args.scope)
    core_types = _parse_core_types(args.core) if scope == "local" else ()
    if scope == "local" and not core_types:
        raise ValueError("When --scope local is used, --core must be provided (for example --core Al,Mg).")
    return DipoleRequest(
        scope=scope,
        atom_types=core_types if scope == "local" else None,
        frames=[int(args.frame)],
    )


def _build_polarization_request(args: argparse.Namespace) -> PolarizationRequest:
    scope = str(args.scope)
    core_types = _parse_core_types(args.core) if scope == "local" else ()
    if scope == "local" and not core_types:
        raise ValueError("When --scope local is used, --core must be provided (for example --core Al,Mg).")
    return PolarizationRequest(
        scope=scope,
        atom_types=core_types if scope == "local" else None,
        frames=[int(args.frame)],
        volume_method="bbox" if scope == "local" else "hull",
    )


def _build_hyst_request(args: argparse.Namespace) -> PolarizationFieldRequest:
    return PolarizationFieldRequest(
        field_component="field_z",
        aggregate=args.aggregate,
        x_variable="field_z",
        y_variable="P_z (uC/cm^2)",
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
    "hyst": _build_hyst_request,
}


def _attach_time_column(data: ElectrostaticsData, table: pd.DataFrame) -> pd.DataFrame:
    simulation = data.trajectory.simulation
    if simulation is None or simulation.time is None:
        return table

    iterations = np.asarray(
        data.trajectory.iterations if data.trajectory.iterations is not None else simulation.iterations,
        dtype=int,
    )
    if iterations.size == 0:
        return table

    time_values = np.asarray(simulation.time, dtype=float)
    if time_values.shape[0] != iterations.shape[0]:
        return table

    aux = pd.DataFrame({"iter": iterations, "time": time_values})
    return table.merge(aux, on="iter", how="left")


def _attach_charge_time_column(data: ChargeData, table: pd.DataFrame, *, control_file: str = "control") -> pd.DataFrame:
    simulation = data.simulation
    if simulation is not None and simulation.time is not None:
        iterations = (
            np.asarray(data.iterations, dtype=int).reshape(-1)
            if data.iterations is not None
            else np.arange(len(np.asarray(simulation.time, dtype=float)), dtype=int)
        )
        time_values = np.asarray(simulation.time, dtype=float)
        if iterations.shape[0] == time_values.shape[0]:
            aux = pd.DataFrame({"iter": iterations, "time": time_values})
            return table.merge(aux, on="iter", how="left")

    if "iter" in table.columns:
        converted, _ = convert_xaxis(table["iter"].to_numpy(dtype=int), "time", control_file=control_file)
        out = table.copy()
        out["time"] = np.asarray(converted, dtype=float)
        return out
    return table


def _summary_text(result) -> str:
    def _fmt(values) -> str:
        if not values:
            return "None found"
        return ", ".join(f"{value:.6g}" for value in values)

    return (
        "Hysteresis Analysis Summary\n"
        "===========================\n\n"
        "Coercive fields (where polarization crosses zero vs field_z)\n"
        "Units: MV/cm\n"
        f"Values: {_fmt(result.polarization_zero_crossings)}\n\n"
        "Remnant polarizations (where field_z crosses zero vs P_z)\n"
        "Units: uC/cm^2\n"
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

    if command != "hyst":
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
    parser.formatter_class = argparse.RawTextHelpFormatter
    _add_runtime_arguments(parser)

    if canonical in {"dipole", "polarization"}:
        parser.description = (
            f"Compute {canonical} data for a single frame.\n\n"
            "Examples:\n"
            f"  reaxkit elect {canonical} --frame 10 --scope total --export {canonical}_frame10.csv\n"
            f"  reaxkit elect {canonical} --frame 10 --scope local --core Al --export local_{canonical}_frame10.csv\n"
            f"  reaxkit elect {canonical} --frame 10 --scope local --core Al --plot plot3d --component z --save {canonical}_plots"
        )
        parser.add_argument("--frame", type=int, required=True, help="0-based frame index in xmolout")
        parser.add_argument("--scope", choices=["total", "local"], default="total", help="Electrostatics scope")
        parser.add_argument("--core", default=None, help="Comma-separated core atom types for local scope")
        _add_scalar_presentation_arguments(parser)
    elif canonical == "charge_table":
        parser.description = (
            "Extract per-atom charges across selected frames.\n\n"
            "Examples:\n"
            "  reaxkit elect charge_table --frames 0:10:2 --export charges.csv\n"
            "  reaxkit elect charge_table --atom-ids 1 2 3 --frames 0,5,10 --plot single\n"
            "  reaxkit elect charge_table --atom-types O H --every 5 --xaxis time --save charge_series.png"
        )
        parser.add_argument("--atom-ids", type=int, nargs="*", default=None, help="1-based atom ids to include")
        parser.add_argument("--atom-types", nargs="*", default=None, help="Element symbols to include")
        parser.add_argument("--frames", default=None, help='Frames: "0,10,20" or "0:100:5"')
        parser.add_argument("--every", type=int, default=1, help="Use every Nth selected frame")
        parser.add_argument("--control", default="control", help="Control file for time-axis conversion")
        _add_table_presentation_arguments(parser)
    elif canonical == "hyst":
        parser.description = (
            "Analyze polarization-field hysteresis behavior.\n\n"
            "Examples:\n"
            "  reaxkit elect hyst --plot --save hysteresis.png\n"
            "  reaxkit elect hyst --xaxis field_z --yaxis pol_z --aggregate mean --export hysteresis.csv\n"
            "  reaxkit elect hyst --roots --summary hysteresis_summary.txt"
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


def _run_charge_table(args: argparse.Namespace) -> int:
    adapter = _resolve_adapter(args)
    data = adapter.load(ChargeData, vars(args))
    request = REQUEST_BUILDERS["charge_table"](args)
    result = ChargeTableTask().run(data, request)
    result.table = _attach_charge_time_column(data, result.table, control_file=getattr(args, "control", "control"))

    if args.export:
        out = resolve_output_path(
            args.export,
            "elect",
            run_id=getattr(args, "run_id", None),
            project_root=getattr(args, "project_root", "."),
            analysis_id=getattr(args, "analysis_id", None),
        )
        args = argparse.Namespace(**vars(args))
        args.export = str(out)

    present_result("charge_table", result, args, plot_payload_builder=_plot_payload)
    if getattr(args, "export", None):
        print(f"[Done] Exported data to {args.export}")
    return 0


def _run_scalar_command(command: str, args: argparse.Namespace) -> int:
    data = _load_electrostatics_data(args)
    request = REQUEST_BUILDERS[command](args)
    task = DipoleTask() if command == "dipole" else PolarizationTask()
    result = task.run(data, request)

    if args.export:
        out = resolve_output_path(
            args.export,
            "elect",
            run_id=getattr(args, "run_id", None),
            project_root=getattr(args, "project_root", "."),
            analysis_id=getattr(args, "analysis_id", None),
        )
        export_result_csv(result, str(out))
        print(f"[Done] Exported data to {out}")

    if not args.plot:
        if not args.export:
            print_result_table(result)
        return 0

    if request.scope != "local":
        raise ValueError(f"--plot {args.plot} requires --scope local.")

    local_table = _local_result_with_coords(data.trajectory, result.table, args.frames)
    payloads = _iter_local_plot_payloads(command, local_table, args)
    for payload in payloads:
        render_plot(payload)

    if args.save:
        if args.plot == "plot3d":
            print(f"[Done] All 3D local {command} plots saved in {args.save}")
        else:
            print(f"[Done] All 2D local {command} heatmaps saved in {args.save}")
    return 0


def _run_hyst(args: argparse.Namespace) -> int:
    data = _load_hysteresis_data(args)
    request = REQUEST_BUILDERS["hyst"](args)
    result = PolarizationFieldTask().run(data, request)
    result.full_table = _attach_time_column(data, result.full_table)

    if args.export:
        out_csv = resolve_output_path(
            args.export,
            "elect",
            run_id=getattr(args, "run_id", None),
            project_root=getattr(args, "project_root", "."),
            analysis_id=getattr(args, "analysis_id", None),
        )
        export_result_csv(SimpleNamespace(table=result.aggregated_table), str(out_csv))
        print(f"[Done] Exported aggregated joint hysteresis data to {out_csv}")

        full_save_path = out_csv.parent / "hysteresis_full_data.csv"
        export_result_csv(SimpleNamespace(table=result.full_table), str(full_save_path))
        print(f"[Done] Exported full hysteresis dataset to {full_save_path}")

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(_summary_text(result), encoding="utf-8")
        print(f"[Done] Wrote hysteresis summary to {summary_path}")

    if args.roots:
        coercive = ", ".join(f"{value:.6g}" for value in result.polarization_zero_crossings) or "None found"
        remnant = ", ".join(f"{value:.6g}" for value in result.field_zero_crossings) or "None found"
        print("\n[Hysteresis roots]")
        print("  Coercive fields (MV/cm):", coercive)
        print("  Remnant polarizations (uC/cm^2):", remnant)

    payload = _plot_payload("hyst", result, args)
    if payload is not None:
        if args.save:
            out_plot = resolve_output_path(
                args.save,
                "elect",
                run_id=getattr(args, "run_id", None),
                project_root=getattr(args, "project_root", "."),
                analysis_id=getattr(args, "analysis_id", None),
            )
            render_plot({**payload, "save": out_plot})
        if args.plot or not args.save:
            render_plot(payload)

    if not args.export and not args.save and not args.plot:
        print(result.aggregated_table.to_string(index=False))

    return 0


def run_main(command: str, args: argparse.Namespace) -> int:
    canonical = resolve_command_name(command, task_names=ELECTROSTATICS_COMMANDS)
    if canonical == "charge_table":
        return _run_charge_table(args)
    if canonical in {"dipole", "polarization"}:
        return _run_scalar_command(canonical, args)
    if canonical == "hyst":
        return _run_hyst(args)
    raise KeyError(f"Unsupported electrostatics command '{canonical}'.")


def _legacy_command_runner(command: str):
    def _runner(args: argparse.Namespace) -> int:
        return run_main(command, args)

    return _runner


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    for command in ELECTROSTATICS_COMMANDS:
        parser = subparsers.add_parser(command, formatter_class=argparse.RawTextHelpFormatter)
        build_parser(parser, command=command)
        parser.set_defaults(_run=_legacy_command_runner(command))
