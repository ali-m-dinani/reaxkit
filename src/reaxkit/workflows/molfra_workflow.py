"""a workflow for getting molfra.out data"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Sequence, Union, List, Mapping, Any
import pandas as pd

from reaxkit.io.molfra_handler import MolFraHandler
from reaxkit.analysis.molfra_analyzer import (
    occurrences_wide,
    qualifying_types,
    largest_molecule_atoms_wide,
)

from reaxkit.utils.frame_utils import parse_frames, select_frames
from reaxkit.utils.convert import convert_xaxis
from reaxkit.analysis.plotter import single_plot
from reaxkit.utils.alias_utils import normalize_choice

FramesT = Optional[Union[slice, Sequence[int]]]

# -----------------
# Internals
# -----------------
def _safe_xcol_name(xlabel: str) -> str:
    """
    Make a CSV-friendly column name from a pretty axis label.
    Examples:
      'iter'     -> 'iter'
      'Frame'         -> 'frame'
      'Time (ps)'     -> 'time_ps'
      'Time (ns)'     -> 'time_ns'
      'Time (fs)'     -> 'time_fs'
    """
    lab = xlabel.strip().lower()
    if lab.startswith("time (") and lab.endswith(")"):
        unit = lab[6:-1].strip()
        return f"time_{unit}"
    return lab.replace(" ", "_").replace("(", "").replace(")", "")

def _maybe_export(df: pd.DataFrame, out_path: Optional[str]) -> None:
    if not out_path:
        return
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[Done] Exported data to {out}")

def _maybe_plot(series_dict: Mapping[str, Sequence[float]],
                x: Sequence[float],
                xlabel: str,
                ylabel: str,
                title: str,
                *,
                save: Optional[str],
                do_plot: bool) -> None:
    """
    Use centralized single_plot. If save is provided, the figure is saved (no GUI).
    If do_plot is False (e.g., export-only), skip plotting entirely.
    """
    if not do_plot and not save:
        return  # neither plot nor save requested
    series = [{"x": x, "y": yvals, "label": name} for name, yvals in series_dict.items()]
    single_plot(series=series, title=title, xlabel=xlabel, ylabel=ylabel, save=save, legend=True)

def _compute_x(iters: Sequence[int], xaxis: str, control_file: str) -> tuple[list[float], str]:
    """
    Convert iter indices to requested x-axis using convert.convert_xaxis.
    """
    x_vals, xlabel = convert_xaxis(iters, xaxis, control_file=control_file)
    return list(x_vals), xlabel

def _default_title(args: argparse.Namespace, molecules: Sequence[str]) -> str:
    if args.threshold is not None and not args.molecules:
        return f"Species with max occurrence ≥ {args.threshold}"
    if len(molecules) == 1:
        return f"{molecules[0]} occurrence"
    return "Molecule occurrence"

# -----------------
# Core GET task
# -----------------
def _molfra_get_task(args: argparse.Namespace) -> int:
    # Normalize xaxis aliases (e.g., 'iter' -> 'iter')
    args.xaxis = normalize_choice(args.xaxis, domain="xaxis") or 'iter'
    handler = MolFraHandler(args.file)
    handler._parse()  # ensure tables present

    # Decide plotting behavior:
    # - If --export only → no plot
    # - If --save given → save only (no interactive window)
    # - If --plot given (with or without save) → do interactive plot (save also if provided)
    do_plot = bool(args.plot) and not bool(args.export)  # exporting alone suppresses plotting

    frames_sel = parse_frames(args.frames)

    # ====================================================
    # A) Totals flags
    # ====================================================
    total_flags = []
    if getattr(args, "total_molecules", False):
        total_flags.append("total_molecules")
    if getattr(args, "total_atoms", False):
        total_flags.append("total_atoms")
    if getattr(args, "total_molecular_mass", False):
        total_flags.append("total_molecular_mass")

    if total_flags:
        if not hasattr(handler, "_df_totals"):
            print("⚠️ Totals not found in handler. Please re-parse with the latest MolFraHandler.")
            return 0

        df_tot = select_frames(handler._df_totals.copy(), frames_sel)
        if df_tot.empty:
            print("⚠️ Totals table empty after frame selection.")
            return 0

        iters = df_tot["iter"].tolist()
        x_vals, xlabel = _compute_x(iters, args.xaxis, args.control)

        available = [c for c in total_flags if c in df_tot.columns]
        if not available:
            print("⚠️ Requested totals not present in the file.")
            return 0
        series = {col: df_tot[col].tolist() for col in available}

        # Export
        if args.export:
            xcol = _safe_xcol_name(xlabel)
            out_df = df_tot[["iter"] + available].copy()
            if args.xaxis != "iter":
                out_df.insert(0, xcol, x_vals)
            _maybe_export(out_df, args.export)

        # Plot/save via centralized plotter
        title = args.title or f"{', '.join(available)} vs {xlabel.lower()}"
        _maybe_plot(series, x_vals, xlabel, "Value", title, save=args.save, do_plot=do_plot)        
        if not args.save and not args.export and not do_plot:
            print("ℹ️ No action selected. Use one or more of --plot, --save, --export.")
        return 0

    # ====================================================
    # B) Largest molecule per-element atom freqs
    # ====================================================
    if args.largest_molecule_atoms:
        df_wide = largest_molecule_atoms_wide(handler)
        if df_wide.empty:
            print("⚠️ No atom data available for largest molecule.")
            return 0

        df_wide = select_frames(df_wide, frames_sel)
        iters = df_wide["iter"].tolist()
        x_vals, xlabel = _compute_x(iters, args.xaxis, args.control)

        element_cols = [c for c in df_wide.columns if c != "iter"]
        if not element_cols:
            print("⚠️ No element columns found to plot.")
            return 0

        series = {elem: df_wide[elem].tolist() for elem in element_cols}

        if args.export:
            xcol = _safe_xcol_name(xlabel)
            out_df = df_wide.copy()
            if args.xaxis != "iter":
                out_df.insert(0, xcol, x_vals)
            _maybe_export(out_df, args.export)

        title = args.title or "Atom freqs (largest molecule)"
        _maybe_plot(series, x_vals, xlabel, "Atom freq", title, save=args.save, do_plot=do_plot)
        if not args.save and not args.export and not do_plot:
            print("ℹ️ No action selected. Use one or more of --plot, --save, --export.")
        return 0

    # ====================================================
    # C) Species occurrences
    # ====================================================
    molecules: Optional[List[str]] = args.molecules
    if not molecules:
        if args.threshold is None:
            print("ℹ️ No molecules or threshold given. Use --molecules ... or --threshold N.")
            return 0
        excl = set(args.exclude or [])
        molecules = qualifying_types(handler, threshold=args.threshold, exclude_types=(excl if excl else None))
        if not molecules:
            print(f"ℹ️ No species reached max occurrence ≥ {args.threshold}.")
            return 0

    wide = occurrences_wide(handler, molecules=molecules, iters=None, by_index=False, fill_value=0)
    if wide.empty:
        print("ℹ️ No data available after selection.")
        return 0

    wide = select_frames(wide, frames_sel)
    iters = wide["iter"].tolist()
    x_vals, xlabel = _compute_x(iters, args.xaxis, args.control)

    series = {m: wide[m].tolist() for m in molecules if m in wide.columns}

    if args.export:
        xcol = _safe_xcol_name(xlabel)
        out_df = wide.copy()
        if args.xaxis != "iter":
            out_df.insert(0, xcol, x_vals)
        _maybe_export(out_df, args.export)

    title = args.title or _default_title(args, molecules)
    _maybe_plot(series, x_vals, xlabel, "freq", title, save=args.save, do_plot=do_plot)

    if not args.save and not args.export and not do_plot:
        print("ℹ️ No action selected. Use one or more of --plot, --save, --export.")
    return 0

# -----------------
# Flags & registry
# -----------------
def _wire_get_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument('--file', default="molfra.out", help='Path to molfra.out')

    # Totals (choose any combination)
    p.add_argument('--total_molecules', action='store_true', help='Plot/export total number of molecules.')
    p.add_argument('--total_atoms', action='store_true', help='Plot/export total number of atoms.')
    p.add_argument('--total_molecular_mass', action='store_true', help='Plot/export total system mass.')

    # Largest-molecule atoms (stable per-element series)
    p.add_argument('--largest-molecule-atoms', action='store_true',
                   help='Plot/export per-element atom freqs for the largest molecule per iter.')

    # Species selection
    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument('--molecules', nargs='+', default=None,
                       help='One or more molecule types to include (e.g., H2O OH N128Al128).')
    group.add_argument('--threshold', type=int, default=None,
                       help='Auto-include all species whose max occurrence ≥ threshold.')
    p.add_argument('--exclude', nargs='*', default=None,
                   help='Species to exclude when using --threshold (e.g., Pt).')

    # Axes & frames
    p.add_argument('--xaxis', default='iter',
                   help="X-axis mode. Canonical: 'iter', 'time', 'frame'. "
                        "Aliases also accepted (e.g., 'iter', 'frm').")
    p.add_argument('--control', default='control',
                   help='Path to control file for time conversion (used when --xaxis time).')
    p.add_argument('--frames', default=None,
                   help="Frame selection (position-based after filtering): 'start:stop[:step]' or 'i,j,k'.")

    # Output controls
    p.add_argument('--title', default=None, help='Custom plot title.')
    p.add_argument('--plot', action='store_true', help='Show the plot interactively.')
    p.add_argument('--save', default=None, help='Save the plot (path or directory).')
    p.add_argument('--export', default=None, help='Export the data table to CSV (path).')

    p.set_defaults(_run=_molfra_get_task)

def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    p_get = subparsers.add_parser(
        'get',
        help=('Get molecule/total occurrence across frames and optionally plot, save, or export. '
              '--> reaxkit molfra get --molecules H2O N128Al128 --plot '
              '--> reaxkit molfra get --threshold 3 --exclude Pt --save species_with_max_occur.png '
              '--> reaxkit molfra get --file molfra_ig.out --total_atoms --total_molecular_mass --export out.csv '
              '--> reaxkit molfra get --largest-molecule-atoms --frames "0:30:2" --xaxis time --save bulk.png')
    )
    _wire_get_flags(p_get)
