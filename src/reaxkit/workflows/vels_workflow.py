# reaxkit/workflows/vels_workflow.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from reaxkit.io.vels_handler import VelsHandler
from reaxkit.analysis.vels_analyzer import vels_get
from reaxkit.utils.plotter import scatter3d_points, heatmap2d_from_3d
from reaxkit.utils.path import resolve_output_path


# ------------------------- small parsers -------------------------

def _parse_atoms_1based(spec: Optional[str]) -> Optional[list[int]]:
    """
    Parse atoms like:
      "1,3,7" or "1 3 7" or "1-5" (inclusive) or "1-10,15,20-25"
    Returns 1-based indices (as stored in vels dfs).
    """
    if spec is None:
        return None
    s = spec.strip()
    if not s:
        return None

    out: list[int] = []
    parts = s.replace(",", " ").split()
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            if a.strip().isdigit() and b.strip().isdigit():
                lo, hi = int(a), int(b)
                if lo <= hi:
                    out.extend(list(range(lo, hi + 1)))
                else:
                    out.extend(list(range(hi, lo + 1)))
        else:
            if p.isdigit():
                out.append(int(p))

    # unique, preserve order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq or None


def _parse_bins(bins: str) -> Union[int, Tuple[int, int]]:
    if "," in bins:
        nx, ny = [int(x) for x in bins.split(",")]
        return (nx, ny)
    return int(bins)


def _value_column_to_key(value_col: str) -> tuple[str, str]:
    """
    Map a requested scalar column to the vels_get key + dataframe column name.
    Examples:
      vx/vy/vz  -> ("velocities", "vx"/"vy"/"vz")
      ax/ay/az  -> ("accelerations", "ax"/"ay"/"az")
      pax/pay/paz -> ("prev_accelerations", "ax"/"ay"/"az")
    """
    v = value_col.strip().lower()
    if v in {"vx", "vy", "vz"}:
        return ("velocities", v)
    if v in {"ax", "ay", "az"}:
        return ("accelerations", v)
    if v in {"pax", "pay", "paz"}:
        # previous accels share ax/ay/az columns
        return ("prev_accelerations", "a" + v[1:])
    raise ValueError("value_col must be one of: vx,vy,vz, ax,ay,az, pax,pay,paz")


# ------------------------- common args -------------------------

def _add_common_vels_io_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--file", default="vels", help="Path to vels/moldyn.vel/molsav file.")
    p.add_argument("--atoms", default=None,
                   help='1-based atom indices (optional). Examples: "1,3,7" or "1-10,25".')
    p.add_argument("--export", default=None, help="Path to export CSV data.")
    p.add_argument("--print", action="store_true", help="Print output to console.")


def _add_common_plot_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--save", default=None, help="Path to save plot image (dir or full filename).")
    p.add_argument("--vmin", type=float, default=None, help="Color scale min (auto if not set).")
    p.add_argument("--vmax", type=float, default=None, help="Color scale max (auto if not set).")
    p.add_argument("--cmap", default="coolwarm", help="Matplotlib colormap.")


# ------------------------- tasks -------------------------

def get_task(args: argparse.Namespace) -> int:
    h = VelsHandler(args.file)

    atoms = _parse_atoms_1based(args.atoms)

    out = vels_get(h, args.key, atoms=atoms)

    # metadata
    if isinstance(out, dict):
        if args.print or (not args.export):
            for k, v in out.items():
                print(f"{k}: {v}")
        if args.export:
            export_path = resolve_output_path(args.export, workflow="vels")
            # simple key/value csv
            dfm = pd.DataFrame([{"key": k, "value": str(v)} for k, v in out.items()])
            dfm.to_csv(export_path, index=False)
            print(f"[Done] Requested data is exported to {export_path}")
        return 0

    # dataframe
    df = out
    if args.print or (not args.export):
        # avoid flooding console; you can change this later
        with pd.option_context("display.max_rows", 30, "display.width", 140):
            print(df)

    if args.export:
        export_path = resolve_output_path(args.export, workflow="vels")
        df.to_csv(export_path, index=False)
        print(f"[Done] Requested data is exported to {export_path}")

    return 0


def plot3d_task(args: argparse.Namespace) -> int:
    h = VelsHandler(args.file)
    atoms = _parse_atoms_1based(args.atoms)

    # coords
    cdf = vels_get(h, "coordinates", atoms=atoms)
    if cdf.empty:
        raise ValueError("No coordinates to plot (empty selection).")

    # scalar values from one of the sections
    key, col = _value_column_to_key(args.value)
    vdf = vels_get(h, key, atoms=atoms)
    if vdf.empty:
        raise ValueError(f"No data for {args.value} to plot (empty selection or missing section).")

    merged = cdf.merge(vdf, on="atom_index", how="inner")
    if merged.empty:
        raise ValueError("No matching atom rows between coordinates and selected values.")

    coords = merged[["x", "y", "z"]].to_numpy(float)
    vals = merged[col].to_numpy(float)

    m = np.isfinite(vals)
    coords = coords[m]
    vals = vals[m]
    if coords.size == 0:
        raise ValueError("All selected values are NaN/invalid; nothing to plot.")

    title = f"{args.value}_3D"
    scatter3d_points(
        coords,
        vals,
        title=title,
        s=args.size,
        alpha=args.alpha,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        elev=args.elev,
        azim=args.azim,
        save=(resolve_output_path(args.save, workflow="vels") if args.save else None),
        show_message=True,
    )
    return 0


def heatmap2d_task(args: argparse.Namespace) -> int:
    h = VelsHandler(args.file)
    atoms = _parse_atoms_1based(args.atoms)

    cdf = vels_get(h, "coordinates", atoms=atoms)
    if cdf.empty:
        raise ValueError("No coordinates to plot (empty selection).")

    key, col = _value_column_to_key(args.value)
    vdf = vels_get(h, key, atoms=atoms)
    if vdf.empty:
        raise ValueError(f"No data for {args.value} to plot (empty selection or missing section).")

    merged = cdf.merge(vdf, on="atom_index", how="inner")
    if merged.empty:
        raise ValueError("No matching atom rows between coordinates and selected values.")

    coords = merged[["x", "y", "z"]].to_numpy(float)
    vals = merged[col].to_numpy(float)
    m = np.isfinite(vals)
    coords = coords[m]
    vals = vals[m]
    if coords.size == 0:
        raise ValueError("All selected values are NaN/invalid; nothing to plot.")

    bins = _parse_bins(args.bins)
    title = f"{args.value}_{args.plane}_heatmap2d"

    heatmap2d_from_3d(
        coords,
        vals,
        plane=args.plane,
        bins=bins,
        agg=args.agg,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap,
        title=title,
        save=(resolve_output_path(args.save, workflow="vels") if args.save else None),
        show_message=True,
    )
    return 0


# ------------------------- CLI registration -------------------------

def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    # ---------------- get ----------------
    g = subparsers.add_parser(
        "get",
        help="Get vels data (coords/vels/accels/prev) for all or selected atoms.",
        description=(
            "Examples:\n"
            "  reaxkit vels get --key velocities --atoms 1,3,7 --print\n"
            "  reaxkit vels get --key coordinates --atoms 1-50 --export coords.csv\n"
            "  reaxkit vels get --key metadata --print\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _add_common_vels_io_args(g)
    g.add_argument("--key", required=True,
                   choices=["metadata", "coordinates", "velocities", "accelerations", "prev_accelerations"],
                   help="Which dataset to return.")
    g.set_defaults(_run=get_task)

    # ---------------- plot3d ----------------
    p3 = subparsers.add_parser(
        "plot3d",
        help="3D scatter: color a scalar (vz, vx, ax, ...) on atomic positions.",
        description=(
            "Examples:\n"
            "  reaxkit vels plot3d --value vz --save vz_3d.png\n"
            "  reaxkit vels plot3d --value vz --atoms 1-500 --cmap coolwarm\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p3.add_argument("--file", default="vels", help="Path to vels/moldyn.vel/molsav file.")
    p3.add_argument("--atoms", default=None,
                    help='1-based atom indices. Examples: "1,3,7" or "1-10,25".')
    p3.add_argument("--value", required=True,
                    help="Scalar to plot: vx,vy,vz, ax,ay,az, pax,pay,paz.")
    _add_common_plot_args(p3)
    p3.add_argument("--size", type=float, default=8.0, help="Marker size.")
    p3.add_argument("--alpha", type=float, default=0.9, help="Marker transparency.")
    p3.add_argument("--elev", type=float, default=22.0, help="3D view elevation.")
    p3.add_argument("--azim", type=float, default=38.0, help="3D view azimuth.")
    p3.set_defaults(_run=plot3d_task)

    # ---------------- heatmap2d ----------------
    h2 = subparsers.add_parser(
        "heatmap2d",
        help="2D heatmap: project positions to xy/xz/yz and bin/aggregate a scalar.",
        description=(
            "Examples:\n"
            "  reaxkit vels heatmap2d --value vz --plane xz --bins 60 --agg mean --save vz_xz.png\n"
            "  reaxkit vels heatmap2d --value vz --plane xy --bins 80,40 --agg max\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    h2.add_argument("--file", default="vels", help="Path to vels/moldyn.vel/molsav file.")
    h2.add_argument("--atoms", default=None,
                    help='1-based atom indices. Examples: "1,3,7" or "1-10,25".')
    h2.add_argument("--value", required=True,
                    help="Scalar to plot: vx,vy,vz, ax,ay,az, pax,pay,paz.")
    h2.add_argument("--plane", default="xy", choices=["xy", "xz", "yz"], help="Projection plane.")
    h2.add_argument("--bins", default="50", help='Grid bins: "N" or "Nx,Ny" (e.g., "80,40").')
    h2.add_argument("--agg", default="mean", help="Aggregation: mean|max|min|sum|count.")
    _add_common_plot_args(h2)
    h2.set_defaults(_run=heatmap2d_task)
