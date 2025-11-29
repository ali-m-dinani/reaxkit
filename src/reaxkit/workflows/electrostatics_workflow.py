"""Workflow for electrostatics (dipole/polarization + hysteresis)"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from reaxkit.io.xmolout_handler import XmoloutHandler
from reaxkit.io.fort7_handler import Fort7Handler
from reaxkit.io.fort78_handler import Fort78Handler
from reaxkit.io.control_handler import ControlHandler

from reaxkit.analysis.electrostatics_analyzer import (
    single_frame_dipoles_polarizations,
    polarization_field_analysis,
)
from reaxkit.analysis.plotter import single_plot

from reaxkit.utils.alias import (
    normalize_choice,
    _resolve_alias,
)
from reaxkit.analysis.electrostatics_analyzer import (
    dipoles_polarizations_over_multiple_frames,
)
from reaxkit.analysis.xmolout_analyzer import get_atom_trajectories
from reaxkit.analysis.plotter import scatter3d_points, heatmap2d_from_3d

from reaxkit.utils.frame_utils import parse_frames, resolve_indices
from reaxkit.utils.alias import resolve_alias_from_columns


# -------------------------------------------------------------------------
# Tasks
# -------------------------------------------------------------------------


def dipole_task(args: argparse.Namespace) -> int:
    """
    reaxkit elect dipole
        --xmolout xmolout --fort7 fort.7
        --frame X --scope {total,local}
        [--core Al,Mg] --export out.csv [--polarization]

    total scope:
        renders total dipole or polarization for a single frame and exports it.

    local scope:
        renders local dipole or polarization for each core atom cluster and exports it.
    """
    # Handlers
    xh = XmoloutHandler(args.xmolout)
    f7 = Fort7Handler(args.fort7)

    # Mode: dipole vs polarization
    mode = "polarization" if args.polarization else "dipole"

    # Scope/core types
    scope = args.scope
    core_types: Optional[list[str]] = None
    if scope == "local":
        if not args.core:
            raise ValueError("When --scope local is used, --core must be provided (e.g. --core Al,Mg).")
        core_types = [c.strip() for c in args.core.split(",") if c.strip()]

    # Compute for the requested frame
    df = single_frame_dipoles_polarizations(
        xh,
        f7,
        frame=args.frame,
        scope=scope,
        core_types=core_types,
        mode=mode,
    )

    # Export
    out_path = Path(args.export)
    df.to_csv(out_path, index=False)
    print(f"\n[Done] Exported data to {out_path}")

    return 0


def hyst_task(args: argparse.Namespace) -> int:
    """
    reaxkit elect hyst
        --xmolout xmolout --fort7 fort.7 --fort78 fort.78 --control control
        [--plot] [--save hyst.png]
        [--yaxis pol_z] [--xaxis time|field_z]
        [--aggregate mean|max|min|last]
        [--export hysteresis.csv]
        [--summary summary.txt]
        [--roots]

    - Uses polarization_field_analysis to build polarization + field dataset.
    - Aggregates according to --aggregate.
    - Exports aggregated joint DataFrame (if --export is given).
    - Writes coercive fields and remnant polarizations to a text file (summary).
    - If --roots is present, also prints these values to the terminal.
    - Makes a hysteresis / time-series plot according to --xaxis/--yaxis
      and plot/save flags, using alias_utils to map names.
    """
    # Handlers
    xh = XmoloutHandler(args.xmolout)
    f7 = Fort7Handler(args.fort7)
    f78 = Fort78Handler(args.fort78)
    ctrl = ControlHandler(args.control)

    # Run basic polarization vs field analysis; keep x/y for roots as field_z vs P_z
    full_df, agg_df, coercive_fields, remnant_pols = polarization_field_analysis(
        xh,
        f7,
        f78,
        ctrl,
        field_var="field_z",
        aggregate=args.aggregate,
        x_variable="field_z",
        y_variable="P_z (uC/cm^2)",
    )

    # ------------------------------------------------------------------
    # Optionally add time information to full_df (for plotting vs time)
    # ------------------------------------------------------------------
    try:
        sim_df = xh.dataframe()
        iter_col = _resolve_alias(sim_df, "iter")
        time_col_name = _resolve_alias(sim_df, "time")
        sim_small = sim_df[[iter_col, time_col_name]].copy()
        sim_small = sim_small.rename(columns={iter_col: "iter", time_col_name: "time"})
        full_df = full_df.merge(sim_small, on="iter", how="left")
    except Exception:
        # If no time is resolvable, we simply skip it; plotting vs time may fail later.
        pass

    # ------------------------------------------------------------------
    # Export aggregated joint DataFrame
    # ------------------------------------------------------------------
    if args.export:
        # Export aggregated data (agg_df)
        out_csv = Path(args.export)
        agg_df.to_csv(out_csv, index=False)
        print(f"\n[Done] Exported aggregated joint hysteresis data to {out_csv}")

        # Export full dataframe in same directory
        full_save_dir = out_csv.parent  # Directory where --export was saved
        full_save_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        full_save_path = full_save_dir / "hysteresis_full_data.csv"
        full_df.to_csv(full_save_path, index=False)
        print(f"[Done] Exported full hysteresis dataset to {full_save_path}\n")

    # ------------------------------------------------------------------
    # Write coercive fields & remnant polarizations to a text file
    # ------------------------------------------------------------------
    summary_path = Path(args.summary or "hysteresis_summary.txt")

    def _fmt_list(values) -> str:
        if not values:
            return "None found"
        return ", ".join(f"{v:.6g}" for v in values)

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Hysteresis Analysis Summary\n")
        f.write("===========================\n\n")
        f.write("Coercive fields (where polarization crosses zero vs field_z)\n")
        f.write("Units: MV/cm\n")
        f.write(f"Values: {_fmt_list(coercive_fields)}\n\n")
        f.write("Remnant polarizations (where field_z crosses zero vs P_z)\n")
        f.write("Units: µC/cm^2\n")
        f.write(f"Values: {_fmt_list(remnant_pols)}\n")

    print(f"[Done] Wrote hysteresis summary to {summary_path}")

    # Also print to terminal if --roots is requested
    if args.roots:
        print("\n[Hysteresis roots]")
        print("  Coercive fields (MV/cm):", _fmt_list(coercive_fields))
        print("  Remnant polarizations (µC/cm^2):", _fmt_list(remnant_pols))

    # ------------------------------------------------------------------
    # Plot: use alias_utils to map --xaxis / --yaxis
    # - If xaxis == time → use per-frame data (full_df)
    # - Else             → use aggregated data (agg_df) for hysteresis
    # ------------------------------------------------------------------
    if args.plot or args.save:
        # Canonical keys
        canonical_x = normalize_choice(args.xaxis or "field_z", domain="xaxis")
        canonical_y = normalize_choice(args.yaxis or "P_z (uC/cm^2)", domain="yaxis")

        # Choose DataFrame for plotting
        if canonical_x == "time":
            df_plot = full_df
        else:
            df_plot = agg_df

        # Resolve actual column names present in df_plot
        x_col = _resolve_alias(df_plot, canonical_x)
        y_col = _resolve_alias(df_plot, canonical_y)

        x = df_plot[x_col]
        y = df_plot[y_col]

        title = f"{y_col} vs {x_col}"
        xlabel = x_col
        ylabel = y_col

        single_plot(x, y, title=title, xlabel=xlabel, ylabel=ylabel, save=args.save, figsize=(6,4))

    return 0

# ------------------------- internals -------------------------

def _indices_from_spec(xh: XmoloutHandler, frames_spec) -> list[int]:
    """Turn a frames spec (slice or list) into concrete frame indices for xh."""
    return resolve_indices(xh, frames=frames_spec, iterations=None, step=None)


def _local_pol_with_coords(
    xh: XmoloutHandler,
    f7: Fort7Handler,
    *,
    core_types: Sequence[str],
    frames_spec,
    mode: str = "polarization",  # or "dipole"
) -> pd.DataFrame:
    """
    Build a DataFrame with local electrostatics + core-atom coordinates:

    columns include:
      frame_index, iter, core_atom_type, core_atom_id,
      x, y, z,
      mu_x (debye), mu_y (debye), mu_z (debye),
      P_x (uC/cm^2), P_y (uC/cm^2), P_z (uC/cm^2), volume (angstrom^3)
    """
    # 1) local dipole/polarization over all frames
    df_local = dipoles_polarizations_over_multiple_frames(
        xh,
        f7,
        scope="local",
        core_types=core_types,
        mode=mode,
    )
    if df_local.empty:
        raise ValueError("No local polarization/dipole data found. Check core types and files.")

    # Restrict to requested frames
    idx_list = _indices_from_spec(xh, frames_spec)
    df_local = df_local[df_local["frame_index"].isin(idx_list)].copy()
    if df_local.empty:
        raise ValueError("No local data in the requested frames.")

    # 2) coordinates of atoms from xmolout (long format)
    traj = get_atom_trajectories(
        xh,
        frames=idx_list,
        every=1,
        atoms=None,
        atom_types=None,
        dims=("x", "y", "z"),
        format="long",
    )
    if traj.empty:
        raise ValueError("No trajectory data returned for requested frames.")

    traj = traj[["frame_index", "atom_id", "x", "y", "z"]].copy()
    traj = traj.rename(columns={"atom_id": "core_atom_id"})

    # 3) join local pol/dipole with core-atom coordinates
    df = pd.merge(
        df_local,
        traj,
        on=["frame_index", "core_atom_id"],
        how="inner",
    )
    if df.empty:
        raise ValueError("Could not match local polarization entries to core atom coordinates.")

    return df


# ========================= 3D SCATTER =========================

def local_pol_plot3d_task(args: argparse.Namespace) -> int:
    """
    3D scatter of local polarization/dipole (one point per core atom) across frames.

    Example:
      reaxkit elect plot3d --core Al,Mg --component pol_z --frames 0:200:20 --save figs/local_pol3d
    """
    xh = XmoloutHandler(args.xmolout)
    f7 = Fort7Handler(args.fort7)

    frames_spec = parse_frames(args.frames)
    core_types = [c.strip() for c in args.core.split(",") if c.strip()]

    df = _local_pol_with_coords(
        xh,
        f7,
        core_types=core_types,
        frames_spec=frames_spec,
        mode="polarization" if args.polarization else "dipole",
    )

    # Resolve which column to color by (mu_z, P_z, etc.) using alias map
    col = resolve_alias_from_columns(df.columns, args.component)
    if col is None:
        raise ValueError(
            f"Component '{args.component}' not found. "
            f"Available columns include: {list(df.columns)[:12]} ..."
        )

    # Output dir
    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Per-frame 3D plot
    for fi in sorted(df["frame_index"].unique()):
        sub = df[df["frame_index"] == fi]
        coords = sub[["x", "y", "z"]].to_numpy(float)
        vals = sub[col].to_numpy(float)

        # skip frames with all-NaN or empty
        if coords.size == 0 or not np.isfinite(vals).any():
            continue

        vmin = args.vmin if args.vmin is not None else float(np.nanmin(vals))
        vmax = args.vmax if args.vmax is not None else float(np.nanmax(vals))

        title = f"{col}_local_3D_frame_{fi}"
        scatter3d_points(
            coords,
            vals,
            title=title,
            s=args.size,
            alpha=args.alpha,
            cmap=args.cmap,
            vmin=vmin,
            vmax=vmax,
            elev=args.elev,
            azim=args.azim,
            save=(save_dir / f"{title}.png" if save_dir else None),
            show_message=False,
        )

    if args.save:
        print(f"[Done] All 3D local polarization plots saved in {args.save}")
    return 0


# ========================= 2D HEATMAP =========================

def _parse_bins(bins: str) -> Union[int, Tuple[int, int]]:
    if "," in bins:
        nx, ny = [int(x) for x in bins.split(",")]
        return (nx, ny)
    return int(bins)


def local_pol_heatmap2d_task(args: argparse.Namespace) -> int:
    """
    2D heatmap of local polarization/dipole on core-atom positions.

    Example:
      reaxkit elect heatmap2d --core Al,Mg --component pol_z --plane xz --bins 40 --frames 0:200:20 --save figs/local_pol_heat
    """
    xh = XmoloutHandler(args.xmolout)
    f7 = Fort7Handler(args.fort7)

    frames_spec = parse_frames(args.frames)
    core_types = [c.strip() for c in args.core.split(",") if c.strip()]

    df = _local_pol_with_coords(
        xh,
        f7,
        core_types=core_types,
        frames_spec=frames_spec,
        mode="polarization" if args.polarization else "dipole",
    )

    col = resolve_alias_from_columns(df.columns, args.component)
    if col is None:
        raise ValueError(
            f"Component '{args.component}' not found. "
            f"Available columns include: {list(df.columns)[:12]} ..."
        )

    bins = _parse_bins(args.bins)

    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for fi in sorted(df["frame_index"].unique()):
        sub = df[df["frame_index"] == fi]
        coords = sub[["x", "y", "z"]].to_numpy(float)
        vals = sub[col].to_numpy(float)

        if coords.size == 0 or not np.isfinite(vals).any():
            continue

        m = np.isfinite(vals)
        coords = coords[m]
        vals = vals[m]

        vmin = args.vmin if args.vmin is not None else float(np.nanmin(vals))
        vmax = args.vmax if args.vmax is not None else float(np.nanmax(vals))

        title = f"{col}_local_{args.plane}_frame_{fi}"
        out_path = (save_dir / f"{title}.png") if save_dir else None

        heatmap2d_from_3d(
            coords,
            vals,
            plane=args.plane,
            bins=bins,
            agg=args.agg,
            vmin=vmin,
            vmax=vmax,
            cmap=args.cmap,
            title=title,
            save=out_path,
            show_message=False,
        )

    if args.save:
        print(f"[Done] All 2D local polarization heatmaps saved in {args.save}")
    return 0


# -------------------------------------------------------------------------
# Registration
# -------------------------------------------------------------------------

def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    # ---------------------- dipole ----------------------
    p_dip = subparsers.add_parser("dipole", help="Compute dipole moment or polarization for a single frame || "
                                                 "reaxkit elect dipole --frame 10 --scope total --export total_frame10.csv --polarization || "
                                                 "reaxkit elect dipole --frame 10 --scope local --core Al --export local_frame10.csv --polarization")
    p_dip.add_argument("--xmolout", default="xmolout", help="Path to xmolout file")
    p_dip.add_argument("--fort7", default="fort.7", help="Path to fort.7 file")
    p_dip.add_argument("--frame", type=int, required=True, help="0-based frame index in xmolout")
    p_dip.add_argument("--scope", choices=["total", "local"], default="total", help="Electrostatics scope: total or local")
    p_dip.add_argument("--core", default=None, help="Comma-separated core atom types for local scope (e.g. Al,Mg)")
    p_dip.add_argument("--export", required=True, help="CSV file to export the dipole/polarization data")
    p_dip.add_argument("--polarization", action="store_true", help="If present, compute polarization instead of dipole")
    p_dip.set_defaults(_run=dipole_task)

    # ---------------------- hysteresis ----------------------
    p_hyst = subparsers.add_parser("hyst", help="Polarization-field hysteresis analysis || "
                                                "reaxkit elect hyst --plot --yaxis pol_z --xaxis field_z --aggregate mean --roots")
    p_hyst.add_argument("--xmolout", default="xmolout", help="Path to xmolout file")
    p_hyst.add_argument("--fort7", default="fort.7", help="Path to fort.7 file")
    p_hyst.add_argument("--fort78", default="fort.78", help="Path to fort.78 file")
    p_hyst.add_argument("--control", default="control", help="Path to control file")
    p_hyst.add_argument("--plot", action="store_true", help="Show the hysteresis or time-series plot")
    p_hyst.add_argument("--save", default='hysteresis_aggregated.png', help="If set, save plot to file (e.g. hyst.png)")
    p_hyst.add_argument("--yaxis", default="pol_z", help="Quantity for y-axis (e.g. pol_z, mu_z, time, P_z)")
    p_hyst.add_argument("--xaxis", default="field_z", help="Quantity for x-axis (e.g. field_z, time, iter)")
    p_hyst.add_argument("--aggregate", choices=["mean", "max", "min", "last"], default="mean", help="Aggregation method")
    p_hyst.add_argument("--export", default='hysteresis_aggregated.csv', help="CSV file to export aggregated hysteresis data")
    p_hyst.add_argument("--summary", default=None, help="Text file to write coercive fields and remnant polarizations")
    p_hyst.add_argument("--roots", action="store_true", help="Also print coercive and remnant values to terminal")
    p_hyst.set_defaults(_run=hyst_task)

    # ---------------------- 3D local polarization scatter ----------------------
    p3d = subparsers.add_parser(
        "plot3d",
        help=(
            "3D scatter of local polarization/dipole on core-atom positions || "
            "reaxkit elect plot3d --core Al --component mu_z --frames 0:3:1 --save figs/local_pol3d"
        ),
    )
    p3d.add_argument("--xmolout", default="xmolout", help="Path to xmolout file")
    p3d.add_argument("--fort7", default="fort.7", help="Path to fort.7 file")
    p3d.add_argument("--core", required=True, help="Comma-separated core atom types (e.g. Al,Mg)")
    p3d.add_argument("--component", default="pol_z",
                     help="Which component to color by (e.g. pol_z, P_z (uC/cm^2), mu_z)")
    p3d.add_argument("--frames", default=None, help='Frames: "0,10,20" or "0:100:5"')
    p3d.add_argument("--polarization", action="store_true",
                     help="Use polarization components (P_x/y/z) instead of dipole")
    p3d.add_argument("--save", default=None, help="Directory to save PNGs (one per frame)")
    p3d.add_argument("--vmin", type=float, default=None, help="Color scale min (auto if not set)")
    p3d.add_argument("--vmax", type=float, default=None, help="Color scale max (auto if not set)")
    p3d.add_argument("--size", type=float, default=20.0, help="Marker size")
    p3d.add_argument("--alpha", type=float, default=0.9, help="Marker transparency")
    p3d.add_argument("--cmap", default="coolwarm", help="Matplotlib colormap")
    p3d.add_argument("--elev", type=float, default=22.0, help="3D view elevation")
    p3d.add_argument("--azim", type=float, default=38.0, help="3D view azimuth")
    p3d.set_defaults(_run=local_pol_plot3d_task)

    # ---------------------- 2D local polarization heatmap ----------------------
    p2d = subparsers.add_parser(
        "heatmap2d",
        help=(
            "2D heatmap of local polarization/dipole on core-atom positions || "
            "reaxkit elect heatmap2d --core Al --component mu_z --plane xz --bins 10 --frames 0:2:1 --save figs/local_pol_heat"
        ),
    )
    p2d.add_argument("--xmolout", default="xmolout", help="Path to xmolout file")
    p2d.add_argument("--fort7", default="fort.7", help="Path to fort.7 file")
    p2d.add_argument("--core", required=True, help="Comma-separated core atom types (e.g. Al,Mg)")
    p2d.add_argument("--component", default="pol_z",
                     help="Which component to aggregate (e.g. pol_z, P_z (uC/cm^2), mu_z)")
    p2d.add_argument("--frames", default=None, help='Frames: "0,10,20" or "0:100:5"')
    p2d.add_argument("--polarization", action="store_true",
                     help="Use polarization components (P_x/y/z) instead of dipole")

    # Projection/grid
    p2d.add_argument("--plane", default="xy", choices=["xy", "xz", "yz"], help="Projection plane")
    p2d.add_argument("--bins", default="40", help='Grid bins: "N" or "Nx,Ny" (e.g., "10,25")')
    p2d.add_argument("--agg", default="mean", help="Aggregation: mean|max|min|sum|count")

    # Viz
    p2d.add_argument("--vmin", type=float, default=None, help="Color scale min (auto if not set)")
    p2d.add_argument("--vmax", type=float, default=None, help="Color scale max (auto if not set)")
    p2d.add_argument("--cmap", default="viridis", help="Matplotlib colormap")
    p2d.add_argument("--save", default=None, help="Directory to save PNGs (one per frame)")
    p2d.set_defaults(_run=local_pol_heatmap2d_task)


