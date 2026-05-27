"""Direct command workflow for spatial per-atom property plots."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import reaxkit.engine  # noqa: F401

from reaxkit.cli.path import resolve_output_path
from reaxkit.core.engine_registry import resolve_engine
from reaxkit.core.command_alias_resolver import resolve_command_name
from reaxkit.core.frame_utils import parse_frame_indices
from reaxkit.core.storage_layout import add_storage_cli_arguments, normalize_storage_args
from reaxkit.domain.data_models import ChargeData, ConnectivityData, TrajectoryData
from reaxkit.presentation.dispatcher import export_result_csv
from reaxkit.presentation.plot import plot as render_plot

ALL_COMMANDS = ("plot_atom_property",)
ALL_LEGACY_COMMANDS = ("atom_property_plot3d", "atom_property_heatmap2d", "plot-atom-property")

_PROPERTY_ALIASES = {
    "charge": "charge",
    "q": "charge",
    "partial_charge": "charge",
    "charges": "charge",
    "sum_bos": "sum_BOs",
    "sum_bo": "sum_BOs",
    "sum_bond_order": "sum_BOs",
    "sum_bond_orders": "sum_BOs",
    "bond_order_sum": "sum_BOs",
    "connectivity": "sum_BOs",
    "count": "count",
}


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None, help="Engine override. Example: --engine reaxff, which applies ReaxFF-specific loading rules.")
    parser.add_argument("--input", default=".", help="Input file or directory for engine resolution. Example: --input runs/job1, which sets base context for file discovery.")
    parser.add_argument("--run-dir", "--dir", dest="run_dir", default=".", help="Run directory fallback for engine detection. Example: --run-dir runs/job1, which acts as backup lookup path.")
    parser.add_argument("--xmolout", default="xmolout", help="Path to trajectory file. Example: --xmolout runs/job1/xmolout, which provides atom coordinates over frames.")
    parser.add_argument("--fort7", default="fort.7", help="Path to fort.7 file. Example: --fort7 runs/job1/fort.7, which provides connectivity/bond-order data.")
    parser.add_argument("--summary", default=None, help="Optional summary.txt path. Example: --summary runs/job1/summary.txt, which supplies auxiliary simulation metadata when available.")
    parser.add_argument("--log", choices=["verbose", "quiet"], default=None, help="Logging level. Example: --log verbose, which prints more runtime details.")
    add_storage_cli_arguments(parser)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--property", default=None, help="Property to map: charge, q, partial_charge, sum_BOs, connectivity. Example: --property charge, which colors atoms by partial charge.")
    parser.add_argument(
        "--frames",
        nargs="*",
        default=None,
        help='Frame selector syntax. Example: --frames 0:20:2, which selects frames 0,2,4,...,20.',
    )
    parser.add_argument("--every", type=int, default=1, help="Use every Nth selected frame. Example: --every 5, which subsamples selected frames by a factor of five.")
    parser.add_argument("--atom-ids", type=int, nargs="*", default=None, help="Restrict to selected 1-based atom ids. Example: --atom-ids 1 2 5, which keeps only those atoms.")
    parser.add_argument("--atom-types", nargs="*", default=None, help="Restrict to selected atom types/elements. Example: --atom-types O H, which keeps only oxygen and hydrogen.")
    parser.add_argument("--save", default=None, help="Directory used when saving frame plots. Example: --save plots3d, which writes one image per frame to that folder.")
    parser.add_argument("--show", action="store_true", help="Show the generated plot windows. Example: --show, which opens plots interactively.")
    parser.add_argument("--export", default=None, help="Export the assembled per-atom table to CSV. Example: --export atom_values.csv, which saves coordinates and mapped values.")
    parser.add_argument("--vmin", type=float, default=None, help="Color scale minimum. Example: --vmin -0.5, which clamps the lower color bound.")
    parser.add_argument("--vmax", type=float, default=None, help="Color scale maximum. Example: --vmax 0.5, which clamps the upper color bound.")
    parser.add_argument("--cmap", default=None, help="Matplotlib colormap. Example: --cmap coolwarm, which sets the visualization color palette.")


def _parse_bins(spec: str) -> int | tuple[int, int]:
    if "," in spec:
        nx, ny = [int(token.strip()) for token in spec.split(",", maxsplit=1)]
        return (nx, ny)
    return int(spec)


def _canonical_property(spec: str | None, *, allow_count: bool = False) -> str:
    if not spec:
        if allow_count:
            return "count"
        raise ValueError("--property is required.")
    key = spec.strip().lower().replace("-", "_").replace(" ", "_")
    canonical = _PROPERTY_ALIASES.get(key)
    if canonical is None or (canonical == "count" and not allow_count):
        valid = sorted(k for k, v in _PROPERTY_ALIASES.items() if allow_count or v != "count")
        raise ValueError(f"Unknown property '{spec}'. Valid aliases include: {valid}")
    return canonical


def _selected_frames(n_frames: int, frames: list[int] | None, every: int) -> list[int]:
    idx = list(range(n_frames)) if frames is None else [int(i) for i in frames]
    return [i for i in idx if 0 <= i < n_frames][:: max(1, int(every))]


def _normalized_args(args: argparse.Namespace) -> dict:
    normalized = normalize_storage_args(vars(args))
    for key, value in normalized.items():
        setattr(args, key, value)
    return normalized


def _detection_path(args_map: dict) -> str:
    for key in ("input", "xmolout", "fort7", "run_dir"):
        value = args_map.get(key)
        if value:
            return str(value)
    return "."


def _resolve_adapter(args: argparse.Namespace):
    args_map = _normalized_args(args)
    return resolve_engine(_detection_path(args_map), engine=getattr(args, "engine", None))


def _load_domain_data(
    args: argparse.Namespace,
    *,
    property_name: str,
) -> tuple[TrajectoryData, ChargeData | None, ConnectivityData | None]:
    adapter = _resolve_adapter(args)
    load_args = _normalized_args(args)
    trajectory = adapter.load(TrajectoryData, load_args)
    charges = adapter.load(ChargeData, load_args) if property_name == "charge" else None
    connectivity = adapter.load(ConnectivityData, load_args) if property_name == "sum_BOs" else None
    return trajectory, charges, connectivity


def _value_matrix(
    *,
    canonical_property: str,
    charges: ChargeData | None,
    connectivity: ConnectivityData | None,
    n_frames: int,
    n_atoms: int,
) -> np.ndarray:
    if canonical_property == "charge":
        if charges is None:
            raise ValueError("ChargeData is required for charge plots.")
        values = np.asarray(charges.charges, dtype=float)
        if values.shape != (n_frames, n_atoms):
            raise ValueError("ChargeData.charges shape must match trajectory (n_frames, n_atoms).")
        return values
    if canonical_property == "sum_BOs":
        if connectivity is None:
            raise ValueError("ConnectivityData is required for connectivity plots.")
        if connectivity.sum_bond_orders is None:
            raise ValueError("ConnectivityData.sum_bond_orders is required for connectivity plots.")
        values = np.asarray(connectivity.sum_bond_orders, dtype=float)
        if values.shape != (n_frames, n_atoms):
            raise ValueError("ConnectivityData.sum_bond_orders shape must match trajectory (n_frames, n_atoms).")
        return values
    if canonical_property == "count":
        return np.ones((n_frames, n_atoms), dtype=float)
    raise KeyError(f"Unsupported property '{canonical_property}'.")


def _assemble_table(
    trajectory: TrajectoryData,
    charges: ChargeData | None,
    connectivity: ConnectivityData | None,
    *,
    property_name: str,
    frames: list[int] | None,
    every: int,
    atom_ids: list[int] | None,
    atom_types: list[str] | None,
) -> pd.DataFrame:
    positions = np.asarray(trajectory.positions, dtype=float)
    if positions.ndim != 3:
        raise ValueError("TrajectoryData.positions must have shape (n_frames, n_atoms, 3).")
    n_frames, n_atoms, _ = positions.shape

    frame_idx = _selected_frames(n_frames, frames, every)
    if not frame_idx:
        return pd.DataFrame(columns=["frame_index", "iter", "atom_id", "atom_type", "x", "y", "z", "value", "property"])

    iterations = (
        np.asarray(trajectory.iterations, dtype=int)
        if trajectory.iterations is not None
        else (
            np.asarray(trajectory.simulation.iterations, dtype=int)
            if trajectory.simulation is not None and trajectory.simulation.iterations is not None
            else np.arange(n_frames, dtype=int)
        )
    )
    if iterations.shape[0] != n_frames:
        raise ValueError("Trajectory iterations length must match number of frames.")

    ids = [int(v) for v in trajectory.atom_ids]
    types = [str(v) for v in trajectory.elements]
    if len(ids) != n_atoms or len(types) != n_atoms:
        raise ValueError("Trajectory atom metadata must match number of atoms.")

    keep = np.ones((n_atoms,), dtype=bool)
    if atom_ids:
        chosen_ids = {int(v) for v in atom_ids}
        keep &= np.asarray([atom_id in chosen_ids for atom_id in ids], dtype=bool)
    if atom_types:
        chosen_types = {str(v) for v in atom_types}
        keep &= np.asarray([atom_type in chosen_types for atom_type in types], dtype=bool)

    values = _value_matrix(
        canonical_property=property_name,
        charges=charges,
        connectivity=connectivity,
        n_frames=n_frames,
        n_atoms=n_atoms,
    )

    rows: list[dict[str, object]] = []
    for fi in frame_idx:
        coords = positions[fi]
        for atom_idx in np.where(keep)[0].tolist():
            rows.append(
                {
                    "frame_index": int(fi),
                    "iter": int(iterations[fi]),
                    "atom_id": ids[atom_idx],
                    "atom_type": types[atom_idx],
                    "x": float(coords[atom_idx, 0]),
                    "y": float(coords[atom_idx, 1]),
                    "z": float(coords[atom_idx, 2]),
                    "value": float(values[fi, atom_idx]),
                    "property": property_name,
                }
            )
    return pd.DataFrame(rows)


def _render_frame_payloads(table: pd.DataFrame, args: argparse.Namespace, *, mode: str) -> None:
    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for frame_index in sorted(table["frame_index"].unique()):
        sub = table[table["frame_index"] == frame_index]
        coords = sub[["x", "y", "z"]].to_numpy(float)
        values = sub["value"].to_numpy(float)
        mask = np.isfinite(values)
        coords = coords[mask]
        values = values[mask]
        if coords.size == 0:
            continue

        vmin = args.vmin if args.vmin is not None else float(np.nanmin(values))
        vmax = args.vmax if args.vmax is not None else float(np.nanmax(values))
        title = f"{sub['property'].iloc[0]}_{mode}_frame_{frame_index}"
        payload: dict[str, object]
        if mode == "plot3d":
            payload = {
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
        else:
            payload = {
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
        if args.save:
            render_plot(payload)
        if args.show or not args.save:
            render_plot({k: v for k, v in payload.items() if k != "save"})


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    _ = command
    parser.set_defaults(command="plot_atom_property")
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Plot per-atom properties in 2D heatmap or 3D scatter mode.\n"
        "Use --type to select the plot style while keeping one unified command.\n\n"
        "Examples:\n"
        "  1. 3D scatter of charges:\n"
        "   reaxkit plot_atom_property --type scatter3d --property charge --frames 0 10 20 --save plots3d\n\n"
        "  2. 2D heatmap of connectivity values:\n"
        "   reaxkit plot_atom_property --type heatmap2d --property sum_BOs --plane xz --bins 60 --save heatmaps\n\n"
        "  3. Export table without plotting:\n"
        "   reaxkit plot_atom_property --type scatter3d --property q --export atom_charge_coords.csv"
    )
    _add_runtime_arguments(parser)
    _add_common_arguments(parser)
    parser.add_argument("--type", required=True, choices=["heatmap2d", "scatter3d"], help="Plot type selector. Example: --type heatmap2d, which projects atom data and aggregates it on a 2D grid.")
    parser.add_argument("--size", type=float, default=8.0, help="Marker size for scatter3d. Example: --size 12, which renders larger point markers.")
    parser.add_argument("--alpha", type=float, default=0.9, help="Marker transparency for scatter3d. Example: --alpha 0.6, which makes points more transparent.")
    parser.add_argument("--elev", type=float, default=22.0, help="3D view elevation for scatter3d. Example: --elev 30, which raises the camera tilt angle.")
    parser.add_argument("--azim", type=float, default=38.0, help="3D view azimuth for scatter3d. Example: --azim 120, which rotates camera around the scene.")
    parser.add_argument("--plane", default="xy", choices=["xy", "xz", "yz"], help="Projection plane for heatmap2d. Example: --plane xz, which projects points onto XZ before binning.")
    parser.add_argument("--bins", default="40", help='Grid bins for heatmap2d: "N" or "Nx,Ny". Example: --bins 80,60, which sets non-square grid resolution.')
    parser.add_argument("--agg", default="mean", help="Aggregation for heatmap2d: mean|max|min|sum|count. Example: --agg max, which stores the maximum value in each bin.")
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    canonical = resolve_command_name(command, task_names=ALL_COMMANDS)
    mode = str(args.type)
    property_name = _canonical_property(getattr(args, "property", None), allow_count=(mode == "heatmap2d"))

    trajectory, charges, connectivity = _load_domain_data(args, property_name=property_name)
    table = _assemble_table(
        trajectory,
        charges,
        connectivity,
        property_name=property_name,
        frames=parse_frame_indices(args.frames),
        every=args.every,
        atom_ids=args.atom_ids,
        atom_types=args.atom_types,
    )

    if args.export:
        out_csv = resolve_output_path(
            args.export,
            canonical,
            run_id=getattr(args, "run_id", None),
            project_root=getattr(args, "project_root", "."),
            analysis_id=getattr(args, "analysis_id", None),
        )
        export_result_csv(SimpleNamespace(table=table), str(out_csv))
        print(f"[Done] Exported data to {out_csv}")

    if args.save:
        out_save = resolve_output_path(
            args.save,
            canonical,
            run_id=getattr(args, "run_id", None),
            project_root=getattr(args, "project_root", "."),
            analysis_id=getattr(args, "analysis_id", None),
        )
        args.save = str(out_save)

    if not table.empty:
        _render_frame_payloads(table, args, mode="plot3d" if mode == "scatter3d" else "heatmap2d")
        if args.save:
            print(f"[Done] Saved plots in {args.save}")
    else:
        print("No data available for plotting.")

    if not args.export and not args.save and not args.show:
        print(table.to_string(index=False))
    return 0
