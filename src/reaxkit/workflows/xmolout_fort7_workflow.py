"""workflow for tasks that need both xmoloout and fort7, such as spatial resolution of charges."""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

# Handlers / analyzers
from reaxkit.io.xmolout_handler import XmoloutHandler
from reaxkit.io.fort7_handler import Fort7Handler
from reaxkit.analysis import fort7_analyzer as f7

# Utils (NEW: replace local parsers with these)
from reaxkit.utils.frame_utils import parse_frames, parse_atoms, resolve_indices

# Aliases (NEW: accept canonical or alias names like 'charge', 'q', etc.)
from reaxkit.utils.alias import resolve_alias_from_columns

# Plotting
from reaxkit.utils.plotter import scatter3d_points, heatmap2d_from_3d


# ------------------------- internals -------------------------

def _indices_from_spec(xh: XmoloutHandler, frames_spec: Optional[Union[slice, Sequence[int]]]) -> list[int]:
    """Turn a frames spec (slice or list) into concrete frame indices for xh."""
    return resolve_indices(xh, frames=frames_spec, iterations=None, step=None)


def _select_atoms(n_atoms: int, atoms: Optional[Sequence[int]]) -> np.ndarray:
    """Return a boolean mask over 0..n_atoms-1 for the selected atom indices."""
    if not atoms:
        return np.ones(n_atoms, dtype=bool)
    mask = np.zeros(n_atoms, dtype=bool)
    for a in atoms:
        if 0 <= int(a) < n_atoms:
            mask[int(a)] = True
    return mask


# ========================= 3D SCATTER (per-frame) =========================

def _plot_fort7_property_3d(
    *,
    xmolout_handler: XmoloutHandler,
    fort7_handler: Fort7Handler,
    property_name: str,
    frames_spec: Optional[Union[slice, Sequence[int]]] = None,
    atoms_list: Optional[Sequence[int]] = None,
    save_dir: Optional[Union[str, Path]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    size: float = 8.0,
    alpha: float = 0.9,
    cmap: str = "coolwarm",
    elev: float = 22.0,
    azim: float = 38.0,
) -> None:
    """
    Plot any scalar property (e.g., partial_charge/charge/q) from fort.7 in 3D across frames.
    """
    # Pull all per-atom data we need from fort7 (tidy: frame_idx, iter, atom_idx, <props...>)
    df_all = f7.get_features_atom(fort7_handler, columns=".*", frames=frames_spec, regex=True)

    # Resolve the requested property via alias map against the columns we actually have
    col = resolve_alias_from_columns(df_all.columns, property_name)
    if col is None:
        raise ValueError(
            f"Column '{property_name}' not found (with alias resolution). "
            f"Available columns include: {list(df_all.columns)[:12]} ..."
        )

    # Auto color limits if not provided
    if vmin is None:
        vmin = float(np.nanmin(df_all[col].to_numpy(dtype=float)))
    if vmax is None:
        vmax = float(np.nanmax(df_all[col].to_numpy(dtype=float)))

    # Prep output
    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Concrete frame indices to iterate
    idx_list = _indices_from_spec(xmolout_handler, frames_spec)

    for fi in idx_list:
        # xmolout: geometry + types
        fr = xmolout_handler.frame(int(fi))
        coords = fr["coords"]                      # (N,3)
        n_atoms = coords.shape[0]
        keep_atoms = _select_atoms(n_atoms, atoms_list)

        # fort7 values aligned by atom_idx
        sub = df_all[df_all["frame_idx"] == fi][["atom_idx", col]].copy()
        vals = np.full(n_atoms, np.nan, dtype=float)
        if not sub.empty:
            idx = sub["atom_idx"].to_numpy(int, copy=False)  # 0-based
            good = (idx >= 0) & (idx < n_atoms)
            vals[idx[good]] = sub[col].to_numpy(float, copy=False)[good]

        # filter & drop NaNs
        use = keep_atoms & np.isfinite(vals)
        if not np.any(use):
            continue

        title = f"{col}_3D_frame_{fi}"
        scatter3d_points(
            coords[use],
            vals[use],
            title=title,
            s=size,
            alpha=alpha,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            elev=elev,
            azim=azim,
            save=(save_dir / f"{title}.png" if save_dir else None),
            show_message = False
        )


def plot_property_task(args: argparse.Namespace) -> int:
    xh = XmoloutHandler(args.xmolout)
    fh = Fort7Handler(args.fort7)

    frames_spec = parse_frames(args.frames)         # NEW
    atoms_list  = parse_atoms(args.atoms)           # NEW

    _plot_fort7_property_3d(
        xmolout_handler=xh,
        fort7_handler=fh,
        property_name=args.property,
        frames_spec=frames_spec,
        atoms_list=atoms_list,
        save_dir=args.save,
        vmin=args.vmin,
        vmax=args.vmax,
        size=args.size,
        alpha=args.alpha,
        cmap=args.cmap,
        elev=args.elev,
        azim=args.azim,
    )
    if args.save:
        print(f"[Done] All plots saved in {args.save}")

    return 0


# ==================== 2D HEATMAP (projection + aggregation) ====================

def _parse_bins(bins: str) -> Union[int, Tuple[int, int]]:
    if "," in bins:
        nx, ny = [int(x) for x in bins.split(",")]
        return (nx, ny)
    return int(bins)


def heatmap2d_task(args: argparse.Namespace) -> int:
    """
    Project 3D coords to 2D (xy/xz/yz), bin into a grid, and aggregate values.
    If --property is provided, aggregate that fort7 scalar; else aggregate point count.
    One PNG per frame is saved if --save is provided; otherwise plots are shown.
    """
    xh = XmoloutHandler(args.xmolout)
    frames_spec = parse_frames(args.frames)         # NEW
    idx_list = _indices_from_spec(xh, frames_spec)

    # Optional scalar from fort7
    use_scalar = args.property is not None
    if use_scalar:
        fh = Fort7Handler(args.fort7)
        df = f7.get_features_atom(fh, columns=".*", frames=frames_spec, regex=True)
        col = resolve_alias_from_columns(df.columns, args.property)
        if col is None:
            raise ValueError(
                f"Column '{args.property}' not found (with alias resolution). "
                f"Available columns include: {list(df.columns)[:12]} ..."
            )

    bins: Union[int, Tuple[int, int]] = _parse_bins(args.bins)

    # Output
    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for fi in idx_list:
        fr = xh.frame(int(fi))
        coords = fr["coords"]

        if use_scalar:
            sub = df[df["frame_idx"] == fi][["atom_idx", col]]
            values = np.full(coords.shape[0], np.nan, float)
            if not sub.empty:
                idx = sub["atom_idx"].to_numpy(int, copy=False)
                ok = (idx >= 0) & (idx < coords.shape[0])
                values[idx[ok]] = sub[col].to_numpy(float, copy=False)[ok]
            m = np.isfinite(values)
            c_plot, v_plot = coords[m], values[m]
            agg = args.agg
            label = col
        else:
            c_plot = coords
            v_plot = np.ones(coords.shape[0], float)
            agg = "count"
            label = "count"

        title = f"{label}_{args.plane}_frame_{fi}"
        out_path = (save_dir / f"{title}.png") if save_dir else None

        heatmap2d_from_3d(
            c_plot,
            v_plot,
            plane=args.plane,
            bins=bins,
            agg=agg,
            vmin=args.vmin,
            vmax=args.vmax,
            cmap=args.cmap,
            title=title,
            save=out_path,
            show_message = False,
        )
    if args.save:
        print(f"[Done] All plots saved in {args.save}")

    return 0


# ==================== CLI registration ====================
def _add_common_xmolout_io_args(
    p: argparse.ArgumentParser,
    *,
    include_plot: bool = False,
) -> None:
    p.add_argument("--xmolout", default="xmolout", help="Path to xmolout file.")
    p.add_argument("--fort7", default="fort.7", help="Path to fort.7 file.")
    if include_plot:
        p.add_argument("--plot", action="store_true", help="Show plot interactively.")

    p.add_argument("--export", default=None, help="Path to export CSV data.")


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    # 3D scatter
    p = subparsers.add_parser(
        "plot3d",
        help="3D scatter plot of any fort7 property (aliases allowed: charge/q → partial_charge).",
        description=(
            "Examples:\n"
            "  reaxkit xmolfort7 plot3d --property charge --frames 0:20:10 \n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _add_common_xmolout_io_args(p, include_plot=True)
    p.add_argument("--property", required=True,
                  help="Column name or alias (e.g., partial_charge, charge, q).")
    p.add_argument("--frames", default=None,
                  help='Frames: "0,10,20" or "0:100:5".')
    p.add_argument("--atoms", default=None,
                  help='Atom indices: "0,1,2" (0-based).')
    p.add_argument("--vmin", type=float, default=None,
                  help="Color scale min (auto if not set).")
    p.add_argument("--vmax", type=float, default=None,
                  help="Color scale max (auto if not set).")
    p.add_argument("--size", type=float, default=8.0, help="Marker size.")
    p.add_argument("--alpha", type=float, default=0.9, help="Marker transparency.")
    p.add_argument("--cmap", default="coolwarm", help="Matplotlib colormap.")
    p.add_argument("--elev", type=float, default=22.0, help="3D view elevation.")
    p.add_argument("--azim", type=float, default=38.0, help="3D view azimuth.")
    p.add_argument("--save", default="reaxkit_outputs/xmol_fort7/3D_scatter/", help="Path to save plot image.")
    p.set_defaults(_run=plot_property_task)





    # 2D heatmap
    q = subparsers.add_parser(
        "heatmap2d",
        help="Project 3D atoms to 2D grid (xy/xz/yz) and aggregate values per cell.",
        description=(
            "Examples:\n"
            "  reaxkit xmolfort7 heatmap2d --property partial_charge --plane xz "
            "--bins 10 --agg mean --frames 0:300:100\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _add_common_xmolout_io_args(q, include_plot=True)
    q.add_argument("--frames", default=None,
                  help='Frames: "0,10,20" or "0:100:5".')

    # Projection/grid
    q.add_argument("--plane", default="xy", choices=["xy", "xz", "yz"],
                  help="Projection plane.")
    q.add_argument("--bins", default="40",
                  help='Grid bins: "N" or "Nx,Ny" (e.g., "10,25").')
    q.add_argument("--agg", default="mean",
                  help="Aggregation: mean|max|min|sum|count.")

    # Optional scalar from fort7
    q.add_argument("--property", default=None,
                  help="fort7 column or alias to aggregate (e.g., partial_charge|charge|q).")

    # Viz
    q.add_argument("--vmin", type=float, default=None,
                  help="Color scale min (auto if not set).")
    q.add_argument("--vmax", type=float, default=None,
                  help="Color scale max (auto if not set).")
    q.add_argument("--cmap", default="viridis", help="Matplotlib colormap.")
    q.add_argument("--save", default="reaxkit_outputs/xmol_fort7/2D_heatmap/", help="Path to save plot image.")
    q.set_defaults(_run=heatmap2d_task)

