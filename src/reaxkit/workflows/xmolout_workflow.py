"""workflow for extracting data from xmolout file to get trajectories, MSD, RDF, etc."""

import argparse
from typing import Optional, Iterable
import numpy as np
import pandas as pd

from reaxkit.io.xmolout_handler import XmoloutHandler
from reaxkit.utils.path import resolve_output_path
from reaxkit.io.xmolout_generator import write_xmolout_from_handler
from reaxkit.utils.plotter import single_plot, multi_subplots
from reaxkit.utils.frame_utils import (
    _select_frames,
    parse_frames as _parse_frames,
    parse_atoms,
)

from reaxkit.analysis.xmolout_analyzer import (
    mean_squared_displacement,
    get_box_dimensions,
    get_atom_trajectories,
)
from reaxkit.analysis.RDF_analyzer import (
    rdf_using_freud,
    rdf_using_ovito,
    rdf_property_over_frames,
)
from reaxkit.utils.convert import convert_xaxis
from reaxkit.utils.units import UNITS

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
def trajget_task(args: argparse.Namespace) -> int:
    xh = XmoloutHandler(args.file)
    frames = _frames_from_args(xh, args)

    # atoms: optional, 1-based → 0-based
    atoms = parse_atoms(args.atoms)
    if atoms is not None:
        atoms = [a - 1 for a in atoms]

    # atom types
    types = _parse_types(args.atom_types)

    # dims
    dims = args.dims or ["x", "y", "z"]
    dims = [d for d in dims if d in ("x", "y", "z")]
    if not dims:
        raise ValueError("At least one of --dims x y z must be provided.")

    # get trajectories
    df = get_atom_trajectories(
        xh,
        frames=frames,
        atoms=atoms,
        atom_types=sorted(types) if types else None,
        dims=dims,
        format=args.format,
    )

    if df.empty:
        print("No trajectory rows selected (check --atoms / --atom-types / frame range).")
        return 1

    # --- X-axis: iter/frame/time from frame_index ---
    if "frame_index" not in df.columns:
        raise KeyError("Expected 'frame_index' column in trajectory data.")
    xvals, xlabel = convert_xaxis(df["frame_index"].to_numpy(), args.xaxis)

    df = df.copy()
    df.insert(0, "xaxis", xvals)

    workflow_name = args.kind
    # --- Export CSV ---
    if args.export:
        out = resolve_output_path(args.export, workflow_name)
        df.to_csv(out, index=False)
        print(f"[Done] Exported trajectories to {out}")

    # --- Plot / Save (only long-format & single dim) ---
    if args.save or args.plot:
        if args.format == "long" and len(dims) == 1:
            dim = dims[0]
            dff = df

            # if multiple atoms, use the first one (as before)
            if "atom_id" in df.columns:
                atom_ids = df["atom_id"].unique().tolist()
                if len(atom_ids) > 1:
                    print(f"ℹ️ Multiple atoms detected: {atom_ids}. Plotting only atom_id={atom_ids[0]}.")
                aid = atom_ids[0]
                dff = df[df["atom_id"] == aid]

            x = dff["xaxis"].values
            y = dff[dim].values
            title = f"{dim} vs {xlabel}"
            if "aid" in locals():
                title += f" (atom={aid})"

            if args.save:
                out = resolve_output_path(args.save, workflow_name)
                single_plot(
                    x, y,
                    title=title,
                    xlabel=xlabel,
                    ylabel=f"{dim} ({UNITS.get(dim, '') or ''})",
                    save=out,
                )

            if args.plot:
                single_plot(
                    x, y,
                    title=title,
                    xlabel=xlabel,
                    ylabel=f"{dim} ({UNITS.get(dim, '') or ''})",
                    save=None,
                )
        else:
            print("⚠️ Plotting is only supported for long-format with a single dim (x|y|z).")

    # --- No action fallback ---
    if not args.export and not args.save and not args.plot:
        print(df.head(10).to_string(index=False))

    return 0

def msd_task(args: argparse.Namespace) -> int:
    xh = XmoloutHandler(args.file)

    if not args.atoms:
        print("❌ --atoms is required (comma/space-separated 1-based indices, e.g. '1,5,12').")
        return 1

    # Parse 1-based atom indices from string (do NOT use parse_atoms here;
    # mean_squared_displacement expects 1-based atoms and handles conversion itself)
    atoms: list[int] = []
    for chunk in str(args.atoms).split(","):
        for tok in chunk.split():
            if tok:
                atoms.append(int(tok))

    if not atoms:
        print("❌ Could not parse any atoms from --atoms.")
        return 1

    # Long-format MSD: frame_index, iter, atom_id, msd  (per-atom, no averaging)
    df_long = mean_squared_displacement(xh, atoms=atoms)
    if df_long.empty:
        print("No MSD data found for the selected atoms.")
        return 1

    workflow_name = args.kind

    # ------------------------------------------------------------------
    # X-axis: iter / frame / time (based on frame_index)
    # ------------------------------------------------------------------
    frames_unique = np.sort(df_long["frame_index"].unique())
    xvals, xlabel = convert_xaxis(frames_unique, args.xaxis)
    frame_to_x = dict(zip(frames_unique, xvals))

    # ------------------------------------------------------------------
    # Export CSV in WIDE format: [xlabel, frame_index, iter, msd[atom]...]
    # ------------------------------------------------------------------
    if args.export:
        wide = (
            df_long
            .pivot(index=["frame_index", "iter"], columns="atom_id", values="msd")
            .sort_index()
        )

        # Rename atom columns: atom_id → msd[atom_id]
        wide.columns = [f"msd[{aid}]" for aid in wide.columns.to_list()]
        wide = wide.reset_index()

        # X column based on chosen xaxis
        frames_wide = wide["frame_index"].to_numpy()
        xvals_wide, xlabel_wide = convert_xaxis(frames_wide, args.xaxis)
        wide.insert(0, xlabel_wide, xvals_wide)

        out = resolve_output_path(args.export, workflow_name)
        wide.to_csv(out, index=False)
        print(f"[Done] MSD wide table saved to {out}")

    # ------------------------------------------------------------------
    # Build plotting data (one series per atom)
    # ------------------------------------------------------------------
    series_combined = []     # for single_plot (all atoms in one axis)
    subplots_data = []       # for multi_subplots (one subplot per atom)

    atom_ids = sorted(df_long["atom_id"].unique())
    for aid in atom_ids:
        dfi = df_long[df_long["atom_id"] == aid].sort_values("frame_index")
        xs = [frame_to_x[i] for i in dfi["frame_index"].to_numpy()]
        ys = dfi["msd"].to_numpy()
        s = {"x": xs, "y": ys, "label": f"atom {aid}"}
        series_combined.append(s)
        subplots_data.append([s])   # each subplot has one series: this atom

    title = f"MSD of atoms: {', '.join(map(str, atom_ids))}"

    # ------------------------------------------------------------------
    # Plot / Save: either single plot or subplots
    # ------------------------------------------------------------------
    if args.save or args.plot:
        if args.subplot:
            # one subplot per atom
            if args.save:
                out = resolve_output_path(args.save, workflow_name)
                multi_subplots(
                    subplots=subplots_data,
                    title=title,
                    xlabel=xlabel,
                    ylabel="Å²",
                    legend=True,
                    save=out,
                )
            if args.plot:
                multi_subplots(
                    subplots=subplots_data,
                    title=title,
                    xlabel=xlabel,
                    ylabel="Å²",
                    legend=True,
                    save=None,
                )
        else:
            # all atoms in a single plot
            if args.save:
                out = resolve_output_path(args.save, workflow_name)
                single_plot(
                    series=series_combined,
                    title=title,
                    xlabel=xlabel,
                    ylabel="Å²",
                    legend=True,
                    save=out,
                )
            if args.plot:
                single_plot(
                    series=series_combined,
                    title=title,
                    xlabel=xlabel,
                    ylabel="Å²",
                    legend=True,
                    save=None,
                )

    # ------------------------------------------------------------------
    # No action fallback
    # ------------------------------------------------------------------
    if not args.export and not args.save and not args.plot:
        print("ℹ️ No action selected. Use --plot, --save, or --export.")
        print(df_long.head(10).to_string(index=False))

    return 0

def rdf_task(args: argparse.Namespace) -> int:
    xh = XmoloutHandler(args.file)
    types_a = _parse_types(args.types_a)
    types_b = _parse_types(args.types_b)
    frames = _frames_from_args(xh, args)
    backend = args.backend.lower()
    workflow_name = args.kind

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
            out = resolve_output_path(args.export, workflow_name)
            df.to_csv(out, index=False)
            print(f"[Done] Exported RDF property table to {out}")

        if args.save or args.plot:
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

            title = f"{args.prop} ({backend}) vs frame"

            if args.save:
                out = resolve_output_path(args.save, workflow_name)
                single_plot(
                    x,
                    y,
                    title=title,
                    xlabel="frame",
                    ylabel=ylabel,
                    save=out,
                )
            if args.plot:
                single_plot(
                    x,
                    y,
                    title=title,
                    xlabel="frame",
                    ylabel=ylabel,
                    save=None,
                )

        if not args.export and not args.save and not args.plot:
            print(df.head(10).to_string(index=False))
        return 0

    # Curve mode (averaged g(r) vs r)
    if backend == "freud":
        r, g = rdf_using_freud(
            xh,
            frames=frames,
            types_a=types_a,
            types_b=types_b,
            r_max=args.r_max if args.r_max is not None else None,
            bins=args.bins,
            average=True,
            return_stack=False,
        )
    elif backend == "ovito":
        r, g = rdf_using_ovito(
            xh,
            frames=frames,
            r_max=(args.r_max if args.r_max is not None else 4.0),
            bins=args.bins,
            types_a=types_a,
            types_b=types_b,
            average=True,
            return_stack=False,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    sel_txt = ""
    if types_a or types_b:
        sa = sorted(types_a) if types_a else ["*"]
        sb = sorted(types_b) if types_b else ["*"]
        sel_txt = f"  A={sa}; B={sb}"
    title = f"RDF [{backend}]{sel_txt}"

    # Only plot if requested
    if args.save or args.plot:
        out = resolve_output_path(args.save, workflow_name)
        if args.save:
            single_plot(
                r,
                g,
                plot_type="line",
                title=title,
                xlabel="r (Å)",
                ylabel="g(r)",
                save=out,
            )
        if args.plot:
            single_plot(
                r,
                g,
                plot_type="line",
                title=title,
                xlabel="r (Å)",
                ylabel="g(r)",
                save=None,
            )

    if args.export:
        out = resolve_output_path(args.export, workflow_name)
        df = pd.DataFrame({"r": r, "g": g})
        df.to_csv(out, index=False)
        print(f"[Done] Exported RDF to {out}")

    if not args.export and not args.save and not args.plot:
        # simple preview if user didn't ask for plot or export
        out = pd.DataFrame({"r": r, "g": g})
        print(out.head(10).to_string(index=False))

    return 0

def boxdims_task(args: argparse.Namespace) -> int:
    xh = XmoloutHandler(args.file)

    # frames: use all if not provided
    frames = _parse_frames(args.frames) if getattr(args, "frames", None) else None

    df = get_box_dimensions(xh, frames=frames)
    if df.empty:
        print("No box-dimension data found for the requested frames.")
        return 1

    # --- X axis: iter / frame / time ---
    xvals, xlabel = convert_xaxis(df["frame_index"].to_numpy(), args.xaxis)
    df = df.copy()
    df.insert(0, "x", xvals)

    workflow_name = args.kind

    # --- Export CSV ---
    if args.export:
        out = resolve_output_path(args.export, workflow_name)
        df.to_csv(out, index=False)
        print(f"[Done] Exported box-dimension table to {out}")

    # --- Plot / Save using multi_subplots ---
    if args.plot or args.save:
        # plot a, b, c if present
        ycols = [c for c in ("a", "b", "c") if c in df.columns]
        if not ycols:
            print("No a/b/c columns found in box-dimension table; nothing to plot.")
        else:
            subplots_data = []
            for col in ycols:
                subplots_data.append([
                    {
                        "x": df["x"].to_numpy(),
                        "y": df[col].to_numpy(),
                        "label": col,
                    }
                ])

            save_target = resolve_output_path(args.save, workflow_name) if args.save else None

            multi_subplots(
                subplots=subplots_data,
                title=f"Box dimensions vs {xlabel}",
                xlabel=xlabel,
                ylabel="Length (Å)",
                legend=True,
                save=save_target if (args.save or args.plot) else None,
            )

    # --- No action fallback ---
    if not args.export and not args.save and not args.plot:
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
def _add_common_xmolout_io_args(
    p: argparse.ArgumentParser,
    *,
    include_plot: bool = False,
) -> None:
    """Common I/O flags for xmolout-based tasks."""
    p.add_argument("--file", default="xmolout", help="Path to xmolout file.")
    if include_plot:
        p.add_argument("--plot", action="store_true", help="Show plot interactively.")
    p.add_argument("--save", default=None, help="Path to save plot image.")
    p.add_argument("--export", default=None, help="Path to export CSV data.")


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    # ------------------------------------------------------------------
    # trajget (single or multi-atom trajectories)
    # ------------------------------------------------------------------
    pt = subparsers.add_parser(
        "trajget",
        help="Get atom trajectories per frame (single or multiple atoms).",
        description=(
            "Get atom trajectories from xmolout.\n\n"
            "Examples:\n"
            "  reaxkit xmolout trajget --atoms 1 --dims z --xaxis time "
            "--save atom1_z.png --export atom1_z.csv\n"
            "  reaxkit xmolout trajget --atom-types Al --dims x y z --format wide "
            "--export Al_all_dims_traj.csv\n"
            "  reaxkit xmolout trajget --atom-types Al --frames 10:200:10 --dims z "
            "--export Al_z_dim_traj.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _add_common_xmolout_io_args(pt, include_plot=True)
    pt.add_argument("--atoms", default=None,
        help="Comma/space separated 1-based atom indices, e.g. '1,5,12'.")
    pt.add_argument("--dims", nargs="+", default=None, choices=["x", "y", "z"],
        help="Coordinate dimensions to include (default: x y z).")
    pt.add_argument("--xaxis", default="frame", choices=["iter", "frame", "time"],
        help="Quantity on x-axis (default: frame).")
    pt.add_argument("--frames", default=None,
        help="Frame selector: 'start:stop[:step]' or 'i,j,k' (default: all).")
    pt.add_argument("--atom-types", "--types", dest="atom_types", default=None,
        help="Comma/space separated atom types, e.g. 'Al,N'.")
    pt.add_argument("--format", choices=["long", "wide"], default="long",
        help="Output table layout: long or wide (default: long).")
    pt.set_defaults(_run=trajget_task)

    # ------------------------------------------------------------------
    # MSD
    # ------------------------------------------------------------------
    p2 = subparsers.add_parser(
        "msd",
        help="Compute mean squared displacement (MSD) for one or more atoms.",
        description=(
            "Compute MSD from xmolout.\n\n"
            "Examples:\n"
            "  reaxkit xmolout msd --atoms 1 --xaxis frame "
            "--save atom1_msd.png --export atom1_msd.csv\n"
            "  reaxkit xmolout msd --atoms 1,2,3 --xaxis time "
            "--save msd.png --export msd.csv\n"
            "  reaxkit xmolout msd --atoms 1,2,3 --subplot "
            "--save msd_subplot.png --export msd_subplot.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _add_common_xmolout_io_args(p2, include_plot=True)
    p2.add_argument("--atoms", required=True,
        help="Comma/space separated 1-based atom indices, e.g. '1,5,12'.")
    p2.add_argument("--xaxis", default="frame", choices=["iter", "frame", "time"],
        help="Quantity on x-axis (default: frame).")
    p2.add_argument("--subplot", action="store_true",
        help="Plot each atom in its own subplot instead of a single combined plot.")
    p2.set_defaults(_run=msd_task)

    # ------------------------------------------------------------------
    # RDF
    # ------------------------------------------------------------------
    p3 = subparsers.add_parser(
        "rdf",
        help="Compute RDF curve or per-frame RDF-derived property.",
        description=(
            "Compute RDF using FREUD or OVITO backends.\n\n"
            "Curve example:\n"
            "  reaxkit xmolout rdf --save rdf.png --export rdf.csv "
            "--frames 0 --bins 200 --r-max 5\n\n"
            "Property example:\n"
            "  reaxkit xmolout rdf --prop area --bins 200 --r-max 5 "
            "--frames 0:10:1 --save rdf_area.png\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _add_common_xmolout_io_args(p3, include_plot=True)
    p3.add_argument("--backend", choices=["freud", "ovito"], default="ovito",
        help="RDF backend: freud or ovito (default: ovito).")
    p3.add_argument(
        "--prop",
        choices=["first_peak", "dominant_peak", "area", "excess_area"],
        default=None,
        help="Compute this RDF-derived property per frame instead of a curve.",
    )
    p3.add_argument("--types-a", "--types_a", dest="types_a", default=None,
        help="Comma/space separated atom types for set A, e.g. 'Al,N'.")
    p3.add_argument("--types-b", "--types_b", dest="types_b", default=None,
        help="Comma/space separated atom types for set B, e.g. 'N'.")
    p3.add_argument("--bins", type=int, default=200, help="Number of RDF bins.")
    p3.add_argument("--r-max", type=float, default=None,
        help="Max radius in Å; default depends on backend.")
    p3.add_argument("--frames", default=None,
        help="Frame selector: 'start:stop[:step]' or 'i,j,k'.")
    p3.add_argument("--every", type=int, default=1,
        help="Use every Nth frame (default: 1).")
    p3.add_argument("--start", type=int, default=None,
        help="First frame index (0-based).")
    p3.add_argument("--stop", type=int, default=None,
        help="Last frame index (0-based).")
    p3.add_argument("--norm", choices=["extent", "cell"], default="extent",
        help="FREUD normalization: extent or cell (default: extent).")
    p3.add_argument("--c-eff", type=float, default=None,
        help="FREUD only: effective c-length for --norm cell.")
    p3.set_defaults(_run=rdf_task)

    # ------------------------------------------------------------------
    # BOX DIMS
    # ------------------------------------------------------------------
    pb = subparsers.add_parser(
        "boxdims",
        help="Get box/cell dimensions per frame and optionally plot them.",
        description=(
            "Get box/cell dimensions from xmolout.\n\n"
            "Examples:\n"
            "  reaxkit xmolout boxdims --frames 0:500:5 --xaxis time "
            "--export box_dims.csv\n"
            "  reaxkit xmolout boxdims --frames 0:100:10 --xaxis iter "
            "--save box_dim_plots.png\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _add_common_xmolout_io_args(pb, include_plot=True)
    pb.add_argument("--frames", default=None,
        help="Frame selector: 'start:stop[:step]' or 'i,j,k'.")
    pb.add_argument("--xaxis", choices=["frame", "iter", "time"], default="frame",
        help="Quantity on x-axis (default: frame).")
    pb.set_defaults(_run=boxdims_task)

    # ------------------------------------------------------------------
    # TRIM
    # ------------------------------------------------------------------
    ptr = subparsers.add_parser(
        "trim",
        help="Write a trimmed copy of xmolout with only atom_type and x,y,z.",
        description=(
            "Trim xmolout to a lighter file with atom type and coordinates only.\n\n"
            "Example:\n"
            "  reaxkit xmolout trim --file xmolout --output xmolout_trimmed\n"
        ),
    )
    ptr.add_argument("--file", default="xmolout",
        help="Input xmolout file.")
    ptr.add_argument("--output", default="xmolout_trimmed",
        help="Output trimmed xmolout file.")
    ptr.set_defaults(_run=trim_task)



