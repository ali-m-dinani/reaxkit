"""Extract data from xmolout file to get trajectories, MSD, RDF, etc."""
import argparse
from typing import Optional, Iterable
import numpy as np
import pandas as pd

from reaxkit.io.xmolout_handler import XmoloutHandler
from reaxkit.io.xmolout_generator import write_xmolout_from_handler
from reaxkit.analysis.plotter import single_plot
from reaxkit.utils.frame_utils import (
    _select_frames,
    parse_frames as _parse_frames,
    parse_atoms,
)

from reaxkit.analysis.xmolout_analyzer import (
    mean_squared_displacement,
    get_box_dimensions,
    get_atom_trajectories,
    get_atom_properties,
    get_atom_type_mapping
)
from reaxkit.analysis.RDF_analyzer import (
    rdf_using_freud,
    rdf_using_ovito,
    rdf_property_over_frames,
)


def _parse_types(s: Optional[str]):
    if s is None or str(s).strip() == "":
        return None
    parts = [p for chunk in str(s).split(",") for p in chunk.split()]
    return set(parts)

def _frames_from_args(xh: XmoloutHandler, args: argparse.Namespace) -> Iterable[int]:
    if getattr(args, "frames", None):
        sel = _parse_frames(args.frames)
        if sel is None:
            try:
                return range(xh.n_frames())
            except Exception:
                return range(len(xh.dataframe()))
        if isinstance(sel, slice):
            try:
                n = xh.n_frames()
            except Exception:
                n = len(xh.dataframe())
            s, e, st = sel.indices(n)
            return range(s, e, st if st != 0 else 1)
        return [int(i) for i in sel]
    return _select_frames(xh, getattr(args, "start", None), getattr(args, "stop", None), getattr(args, "every", 1))


# --------------------------
# TASK IMPLEMENTATIONS
# --------------------------

def single_atom_traj_task(args: argparse.Namespace) -> int:
    xh = XmoloutHandler(args.file)
    # 1-based -> 0-based for trajectory selector
    atom_zero = args.atom - 1
    df = get_atom_trajectories(xh, atoms=[atom_zero], dims=(args.dim,), format="long")
    if df.empty:
        print("No data for the requested atom/dimension.")
        return 1
    x = df["frame_index"].to_numpy()
    y = df[args.dim].to_numpy()
    single_plot(x, y,
        title=f"Atom {args.atom} {args.dim}-trajectory",
        xlabel="frame", ylabel=args.dim, save=args.save)
    return 0


def msd_task(args: argparse.Namespace) -> int:
    xh = XmoloutHandler(args.file)
    df = mean_squared_displacement(xh, atoms=[args.atom])  # 1-based input is correct
    x = df["frame_index"].to_numpy()
    y = df["msd"].to_numpy()
    single_plot(x, y, title=f"Atom {args.atom} MSD", xlabel="frame", ylabel="MSD", save=args.save)
    return 0


def rdf_task(args: argparse.Namespace) -> int:
    xh = XmoloutHandler(args.file)
    types_a = _parse_types(args.types_a)
    types_b = _parse_types(args.types_b)
    frames = _frames_from_args(xh, args)
    backend = args.backend.lower()

    # Property mode
    if args.prop is not None:
        df = rdf_property_over_frames(
            xh,
            frames=frames,
            backend=backend,
            property=args.prop,  # one of: first_peak, dominant_peak, area, excess_area
            r_max=args.r_max,
            bins=args.bins,
            types_a=sorted(types_a) if types_a else None,
            types_b=sorted(types_b) if types_b else None,
        )
        if args.export:
            df.to_csv(args.export, index=False)
            print(f"📄 Exported RDF property table to {args.export}")
        if args.save:
            x = df["frame_index"].to_numpy()
            # pick the right y-column
            if args.prop == "area":
                y = df["area"].to_numpy()
                ylabel = "area"
            elif args.prop == "excess_area":
                y = df["excess_area"].to_numpy()
                ylabel = "excess_area"
            elif args.prop == "first_peak":
                y = df["r_first_peak"].to_numpy()
                ylabel = "r_first_peak (Å)"
            else:  # dominant_peak
                y = df["r_peak"].to_numpy()
                ylabel = "r_peak (Å)"
            single_plot(x, y,
                        title=f"{args.prop} ({backend}) vs frame", xlabel="frame", ylabel=ylabel, save=args.save)
        if not args.export and not args.save:
            print(df.head(10).to_string(index=False))
        return 0

    # Curve mode (averaged)
    if backend == "freud":
        r, g = rdf_using_freud(
            xh, frames=frames, types_a=types_a, types_b=types_b,
            r_max=args.r_max if args.r_max is not None else None,
            bins=args.bins, average=True, return_stack=False
        )
    elif backend == "ovito":
        r, g = rdf_using_ovito(
            xh, frames=frames, r_max=(args.r_max if args.r_max is not None else 4.0),
            bins=args.bins, types_a=types_a, types_b=types_b, average=True, return_stack=False
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    sel_txt = ""
    if types_a or types_b:
        sa = sorted(types_a) if types_a else ["*"]
        sb = sorted(types_b) if types_b else ["*"]
        sel_txt = f"  A={sa}; B={sb}"
    title = f"RDF [{backend}]{sel_txt}"

    single_plot(
        r,
        g,
        plot_type="line",
        title=title,
        xlabel="r (Å)",
        ylabel="g(r)",
        save=args.save,
    )
    if args.export:
        out = pd.DataFrame({"r": r, "g": g})
        out.to_csv(args.export, index=False)
        print(f"📄 Exported RDF to {args.export}")
    return 0


def boxdims_task(args: argparse.Namespace) -> int:
    xh = XmoloutHandler(args.file)
    frames = _frames_from_args(xh, args)
    df = get_box_dimensions(xh, frames=frames)

    if args.export:
        df.to_csv(args.export, index=False)
        print(f"📄 Exported box dimensions to {args.export}")

    if args.save_prefix:
        for col in ("a", "b", "c"):
            if col in df.columns:
                x = df["frame_index"].values
                y = df[col].values
                single_plot(
                    x, y,
                    title=f"{col}(frame)",
                    xlabel="frame",
                    ylabel=col,
                    save=f"{args.save_prefix}_{col}.png",
                )

    if not args.export and not args.save_prefix:
        print(df.head(10).to_string(index=False))
    return 0


def multi_atom_trajget_task(args: argparse.Namespace) -> int:
    xh = XmoloutHandler(args.file)
    frames = _frames_from_args(xh, args)
    atoms = parse_atoms(args.atoms)
    if atoms is not None:
        atoms = [a - 1 for a in atoms]  # user-facing 1-based -> internal 0-based
    types = _parse_types(args.atom_types)

    dims = args.dims or ["x", "y", "z"]
    dims = [d for d in dims if d in ("x", "y", "z")]
    if not dims:
        raise ValueError("At least one of --dims x y z must be provided.")

    df = get_atom_trajectories(
        xh,
        frames=frames,
        atoms=atoms,
        atom_types=sorted(types) if types else None,
        dims=dims,
        format=args.format,
    )

    if args.export:
        df.to_csv(args.export, index=False)
        print(f"📄 Exported trajectories to {args.export}")

    if args.save:
        if args.format == "long" and len(dims) == 1:
            dim = dims[0]
            if "atom_id" in df.columns:
                atom_ids = df["atom_id"].unique().tolist()
                if len(atom_ids) > 1:
                    print(f"ℹ️ Multiple atoms detected: {atom_ids}. Plotting only atom_id={atom_ids[0]}.")
                aid = atom_ids[0]
                dff = df[df["atom_id"] == aid]
            else:
                dff = df
            x = dff["frame_index"].values
            y = dff[dim].values
            single_plot(
                x, y,
                title=f"{dim}(frame){f' atom={aid}' if 'aid' in locals() else ''}",
                xlabel="frame",
                ylabel=dim,
                save=args.save,
            )
            print(f"[Done] Saved trajectory plot to {args.save}")
        else:
            print("⚠️ Skipping plot: only supported for long-format with a single dim (x|y|z).")

    if not args.export and not args.save:
        print(df.head(10).to_string(index=False))
    return 0


def zspan_task(args: argparse.Namespace) -> int:
    xh = XmoloutHandler(args.file)
    frames = _frames_from_args(xh, args)
    atoms = parse_atoms(args.atoms)
    if atoms is not None:
        atoms = [a - 1 for a in atoms]  # user-facing 1-based -> internal 0-based
    types = _parse_types(args.atom_types)

    traj = get_atom_trajectories(
        xh,
        frames=frames,
        atoms=atoms,
        atom_types=sorted(types) if types else None,
        dims=("z",),
        format="long",
    )

    if traj.empty:
        print("No trajectory rows selected (check --atoms / --atom-types / frame range).")
        return 1

    p_low, p_high = args.percentiles
    if not (0.0 <= p_low < p_high <= 100.0):
        raise ValueError("--percentiles must satisfy 0 <= low < high <= 100")

    def _span(g):
        z = g["z"].to_numpy()
        z_lo = np.nanpercentile(z, p_low)
        z_hi = np.nanpercentile(z, p_high)
        return pd.Series(
            {
                "iter": int(g["iter"].iloc[0]) if "iter" in g else int(g.name),
                "z_min": float(z_lo if (p_low > 0.0) else np.nanmin(z)),
                "z_max": float(z_hi if (p_high < 100.0) else np.nanmax(z)),
                "z_span": float(z_hi - z_lo) if (p_low > 0.0 or p_high < 100.0) else float(np.nanmax(z) - np.nanmin(z)),
            }
        )

    df = traj.groupby("frame_index", as_index=True).apply(_span).reset_index().sort_values("frame_index")

    if args.export:
        df.to_csv(args.export, index=False)
        print(f"📄 Exported z-span table to {args.export}")

    if args.save:
        x = df["frame_index"].to_numpy()
        y = df["z_span"].to_numpy()
        title = f"z-span (p{p_low:g}–p{p_high:g}) vs frame" if (p_low > 0.0 or p_high < 100.0) else "z_max - z_min vs frame"
        single_plot(
            x, y,
            title=title,
            xlabel="frame",
            ylabel="z_span (Å)",
            save=args.save,
        )

    if not args.export and not args.save:
        print(df.head(10).to_string(index=False))

    return 0


def extras_task(args: argparse.Namespace) -> int:
    xh = XmoloutHandler(args.file)
    frames = _frames_from_args(xh, args)
    atoms = parse_atoms(args.atoms)
    if atoms is not None:
        atoms = [a - 1 for a in atoms]  # user-facing 1-based -> internal 0-based
    types = _parse_types(args.atom_types)

    extra_cols = None
    if args.columns and args.columns.lower() != "all":
        extra_cols = [c.strip() for c in args.columns.replace(",", " ").split() if c.strip()]

    df = get_atom_properties(
        xh,
        frames=frames,
        every=max(1, int(getattr(args, "every", 1))),
        atoms=atoms,
        atom_types=sorted(types) if types else None,
        extra_cols=extra_cols,
        include_xyz=args.include_xyz,
        format=args.format,
    )

    if args.export:
        df.to_csv(args.export, index=False)
        print(f"📄 Exported extras to {args.export}")

    # simple plotting: one column vs frame for a single atom (if long format and single col)
    if args.save and args.format == "long":
        value_cols = [c for c in df.columns if c not in ("frame_index", "iter", "atom_id", "atom_type")]
        if len(value_cols) == 1:
            col = value_cols[0]
            if "atom_id" in df.columns:
                atom_ids = df["atom_id"].unique().tolist()
                aid = atom_ids[0]
                if len(atom_ids) > 1:
                    print(f"ℹ️ Multiple atoms detected: {atom_ids}. Plotting only atom_id={aid}.")
                dff = df[df["atom_id"] == aid]
            else:
                dff = df
            x = dff["frame_index"].to_numpy()
            y = dff[col].to_numpy()
            single_plot(
                x, y,
                title=f"{col}(frame){f' atom={aid}' if 'aid' in locals() else ''}",
                xlabel="frame",
                ylabel=col,
                save=args.save,
            )
            print(f"[Done] Saved extras plot to {args.save}")
        else:
            print("⚠️ Skipping plot: provide exactly one value column (use --columns) in long format.")
    elif args.save and args.format != "long":
        print("⚠️ Skipping plot: only supported for long format.")

    if not args.export and not args.save:
        print(df.head(10).to_string(index=False))
    return 0

def trim_task(args: argparse.Namespace) -> int:
    """
    Write a trimmed copy of an xmolout file that keeps only atom_type and x,y,z
    columns for each atom line (drops any extra per-atom columns).
    """
    xh = XmoloutHandler(args.file)
    out_path = args.output or "xmolout_trimmed"
    # include_extras=False ⇒ only atom_type + x y z
    write_xmolout_from_handler(xh, out_path, include_extras=False)
    print(f"[Done] Wrote trimmed xmolout (type + x,y,z only) to {out_path}")
    return 0

# --------------------------
# CLI REGISTRATION
# --------------------------

def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    # trajectory plot
    # straj
    p = subparsers.add_parser(
        "straj",
        help="Plot a single atom's trajectory.\n"
             "reaxkit xmolout straj --atom 1 --dim z --save atom1_z.png"
    )
    p.add_argument("--file", default="xmolout")
    p.add_argument("--atom", type=int, required=True)
    p.add_argument("--dim", choices=["x", "y", "z"], default="z")
    p.add_argument("--save", default=None)
    p.set_defaults(_run=single_atom_traj_task)

    # mtraj
    pt = subparsers.add_parser(
        "mtraj",
        help=(
            "Get atom trajectories per frame (long or wide format).\n"
            "Examples:\n"
            "  reaxkit xmolout mtraj --atom-types Al,N --dims x y z --format wide --export traj.csv\n"
            "  reaxkit xmolout mtraj --frames 10:200:10 --dims z --export z_traj.csv"
        )
    )
    pt.add_argument("--file", default="xmolout")
    pt.add_argument("--frames", default=None,
                    help="Frame selector: 'start:stop[:step]' or 'i,j,k'. Overrides --start/--stop/--every.")
    pt.add_argument("--every", type=int, default=1, help="Use every Nth frame")
    pt.add_argument("--start", type=int, default=None, help="First frame index (0-based)")
    pt.add_argument("--stop", type=int, default=None, help="Last frame index (0-based, inclusive)")
    pt.add_argument("--atoms", default=None, help="Comma/space separated atom indices, e.g., '0,5,12'")
    pt.add_argument("--atom-types", "--types", dest="atom_types", default=None,
                    help="Comma/space separated atom types, e.g., 'Al,N'")
    pt.add_argument("--dims", nargs="+", default=["x", "y", "z"], choices=["x", "y", "z"],
                    help="Coordinate dimensions to include")
    pt.add_argument("--format", choices=["long", "wide"], default="long", help="Output table layout")
    pt.add_argument("--export", default=None, help="CSV output path")
    pt.add_argument("--save", default=None, help="Optional: save a simple plot (only long-format & single dim)")
    pt.set_defaults(_run=multi_atom_trajget_task)

    # MSD
    p2 = subparsers.add_parser(
        "msd",
        help="Plot mean squared displacement (MSD) of a single atom.\n"
             "reaxkit xmolout msd --atom 1 --save atom1_msd.png"
    )
    p2.add_argument("--file", default="xmolout")
    p2.add_argument("--atom", type=int, required=True)
    p2.add_argument("--save", default=None)
    p2.set_defaults(_run=msd_task)

    # RDF
    p3 = subparsers.add_parser(
        "rdf",
        help=(
            "RDF curve or per-frame RDF property using FREUD/OVITO backends.\n"
            "Curve example:\n"
            "  reaxkit xmolout rdf --save rdf.png --export rdf.csv --frames 0 --bins 200 --r-max 5\n"
            "Property example:\n"
            "  reaxkit xmolout rdf --prop area --bins 200 --r-max 5 --frames 0:10:1 --save rdf_area.png"
        )
    )
    p3.add_argument("--file", default="xmolout", help="Path to xmolout")
    p3.add_argument("--backend", choices=["freud", "ovito"], default="ovito", help="RDF backend")
    p3.add_argument("--prop", choices=["first_peak", "dominant_peak", "area", "excess_area"], default=None,
                    help="If set, compute this RDF-derived property per frame instead of the RDF curve")
    p3.add_argument("--types-a", "--types_a", dest="types_a", default=None,
                    help="Comma/space-separated atom types for set A (e.g., 'Al,N')")
    p3.add_argument("--types-b", "--types_b", dest="types_b", default=None,
                    help="Comma/space-separated atom types for set B (e.g., 'N')")
    p3.add_argument("--bins", type=int, default=200, help="Number of RDF bins")
    p3.add_argument("--r-max", type=float, default=None, help="Max radius (Å); default depends on backend")
    # unified frame options
    p3.add_argument("--frames", default=None,
                    help="Frame selector: 'start:stop[:step]' or 'i,j,k'. Overrides --start/--stop/--every.")
    p3.add_argument("--every", type=int, default=1, help="Use every Nth frame")
    p3.add_argument("--start", type=int, default=None, help="First frame index (0-based)")
    p3.add_argument("--stop", type=int, default=None, help="Last frame index (0-based, inclusive)")
    # output
    p3.add_argument("--save", default=None, help="Path to save plot")
    p3.add_argument("--export", default=None, help="Path to export CSV")
    # FREUD-only normalization options
    p3.add_argument("--norm", choices=["extent", "cell"], default="extent",
                    help="FREUD only: 'extent' uses per-frame xyz extents; 'cell' uses a*b*c_eff")
    p3.add_argument("--c-eff", type=float, default=None,
                    help="FREUD only: effective slab thickness (Å) if --norm=cell")
    p3.set_defaults(_run=rdf_task)

    # BOX DIMS
    pb = subparsers.add_parser(
        "boxdims",
        help="Get box/cell dimensions per frame.\n"
             "Example:\n"
             "  reaxkit xmolout boxdims --frames 0:500:5 --save-prefix box"
    )
    pb.add_argument("--file", default="xmolout")
    pb.add_argument("--frames", default=None,
                    help="Frame selector: 'start:stop[:step]' or 'i,j,k'. Overrides --start/--stop/--every.")
    pb.add_argument("--every", type=int, default=1, help="Use every Nth frame")
    pb.add_argument("--start", type=int, default=None, help="First frame index (0-based)")
    pb.add_argument("--stop", type=int, default=None, help="Last frame index (0-based, inclusive)")
    pb.add_argument("--export", default=None, help="CSV output path")
    pb.add_argument("--save-prefix", default=None,
                    help="If given, saves a/b/c vs frame as separate PNGs with this prefix")
    pb.set_defaults(_run=boxdims_task)

    # Z-SPAN
    pz = subparsers.add_parser(
        "zspan",
        help=(
            "Compute and plot z_max - z_min (or robust percentile span) per frame.\n"
            "Examples:\n"
            "  reaxkit xmolout zspan --file xmolout --every 5 --save zspan.png --export zspan.csv\n"
            "  reaxkit xmolout zspan --file xmolout --frames 0:500 --percentiles 5 95 --export zspan_robust.csv"
        )
    )
    pz.add_argument("--file", default="xmolout")
    pz.add_argument("--frames", default=None,
                    help="Frame selector: 'start:stop[:step]' or 'i,j,k'. Overrides --start/--stop/--every.")
    pz.add_argument("--every", type=int, default=1, help="Use every Nth frame")
    pz.add_argument("--start", type=int, default=None, help="First frame index (0-based)")
    pz.add_argument("--stop", type=int, default=None, help="Last frame index (0-based, inclusive)")
    pz.add_argument("--atoms", default=None, help="Comma/space separated atom indices, e.g., '0,5,12'")
    pz.add_argument("--atom-types", "--types", dest="atom_types", default=None,
                    help="Comma/space separated atom types, e.g., 'Al,N'")
    pz.add_argument("--percentiles", nargs=2, type=float, metavar=("PLOW","PHIGH"),
                    default=(0.0, 100.0),
                    help="Robust span using percentiles (default 0 100 equals min–max)")
    pz.add_argument("--export", default=None, help="CSV output path for z-span table")
    pz.add_argument("--save", default=None, help="Save a plot of z-span vs frame to this path")
    pz.set_defaults(_run=zspan_task)

    # TRIM
    ptr = subparsers.add_parser(
        "trim",
        help=(
            "Write a trimmed copy of xmolout with only atom_type and x,y,z per atom line.\n"
            "Example:\n"
            "  reaxkit xmolout trim --file xmolout --output xmolout_trimmed"
        )
    )
    ptr.add_argument("--file", default="xmolout", help="Input xmolout file")
    ptr.add_argument("--output", default="xmolout_trimmed", help="Output trimmed xmolout file")
    ptr.set_defaults(_run=trim_task)



    # NEW: extras
    pe = subparsers.add_parser(
        "extras",
        help=(
            "Export per-atom extra columns preserved in xmolout (beyond x,y,z).\n"
            "Examples:\n"
            "  reaxkit xmolout extras --file xmolout --columns all --export extras.csv\n"
            "  reaxkit xmolout extras --file xmolout --frames 0:500:5 --atoms 0,1,2 --columns q vx vy vz --export evo.csv\n"
            "  reaxkit xmolout extras --file xmolout --atom-types Al --columns unknown_1 --format long --save u1.png"
        )
    )
    pe.add_argument("--file", default="xmolout", help="Path to xmolout")
    pe.add_argument("--frames", default=None,
                    help="Frame selector: 'start:stop[:step]' or 'i,j,k'. Overrides --start/--stop/--every.")
    pe.add_argument("--every", type=int, default=1, help="Use every Nth frame")
    pe.add_argument("--start", type=int, default=None, help="First frame index (0-based)")
    pe.add_argument("--stop", type=int, default=None, help="Last frame index (0-based, inclusive)")
    pe.add_argument("--atoms", default=None, help="Comma/space separated atom indices, e.g., '0,5,12'")
    pe.add_argument("--atom-types", "--types", dest="atom_types", default=None,
                    help="Comma/space separated atom types, e.g., 'Al,N'")
    pe.add_argument("--columns", default="all",
                    help="Which extra columns to include. 'all' or a space/comma list (e.g., 'q vx vy vz' or 'unknown_1').")
    pe.add_argument("--include-xyz", action="store_true",
                    help="Also include x,y,z in addition to the requested extra columns")
    pe.add_argument("--format", choices=["long", "wide"], default="long", help="Output table layout")
    pe.add_argument("--export", default=None, help="CSV output path")
    pe.add_argument("--save", default=None, help="Optional: save a simple plot (only long-format & single value col)")
    pe.set_defaults(_run=extras_task)