"""
Molecular-fragment (molfra) analysis workflow for ReaxKit.

This workflow provides tools for analyzing ReaxFF `molfra.out` and
`molfra_ig.out` files, which describe molecular fragments, species
identities, and their evolution during a simulation.

It supports:
- Tracking occurrences of selected molecular species across frames,
  with optional automatic selection based on occurrence thresholds.
- Computing and visualizing global totals, including total number of
  molecules, total atoms, and total molecular mass versus iteration,
  frame, or physical time.
- Identifying and analyzing the largest molecule in the system, either
  by individual molecular mass or by per-element atom composition.
- Plotting results, saving figures, and exporting processed data to CSV
  using standardized ReaxKit output paths.

The workflow is designed to enable systematic analysis of chemical
speciation, fragmentation, and growth processes in ReaxFF simulations.
"""


from __future__ import annotations

import argparse
from typing import Optional, Sequence, Union, List

from reaxkit.io.handlers.molfra_handler import MolFraHandler
from reaxkit.analysis.per_file.molfra_analyzer import (
    get_molfra_data_wide_format,
    _qualifying_types,
    get_molfra_totals_vs_axis,
    largest_molecule_by_individual_mass,
    atoms_in_the_largest_molecule_wide_format,
)
from reaxkit.utils.frame_utils import parse_frames, select_frames
from reaxkit.utils.media.convert import convert_xaxis
from reaxkit.utils.media.plotter import single_plot, multi_subplots
from reaxkit.utils.alias import normalize_choice
from reaxkit.utils.path import resolve_output_path

FramesT = Optional[Union[slice, Sequence[int]]]


# ============================================================
# Occurrences task
# ============================================================
def _molfra_occur_task(args: argparse.Namespace) -> int:
    # Normalize x-axis alias
    args.xaxis = normalize_choice(args.xaxis, domain="xaxis") or "iter"

    handler = MolFraHandler(args.file)
    handler._parse()

    # Plot behavior: export-only suppresses interactive plotting
    do_plot = bool(args.plot) and not bool(args.export)

    frames_sel = parse_frames(args.frames)

    # -------------------------
    # Determine which molecules
    # -------------------------
    molecules: Optional[List[str]] = args.molecules
    if not molecules:
        if args.threshold is None:
            print("ℹ️ No molecules or threshold given. Use --molecules ... or --threshold N.")
            return 0
        excl = set(args.exclude or [])
        molecules = _qualifying_types(
            handler,
            threshold=args.threshold,
            exclude_types=(excl if excl else None),
        )
        if not molecules:
            print(f"ℹ️ No species reached max occurrence ≥ {args.threshold}.")
            return 0

    wide = get_molfra_data_wide_format(
        handler,
        molecules=molecules,
        iters=None,
        by_index=False,
        fill_value=0,
    )
    if wide.empty:
        print("ℹ️ No data available after selection.")
        return 0

    wide = select_frames(wide, frames_sel)

    if wide.empty:
        print("ℹ️ No data left after frame selection.")
        return 0

    # -------------------------
    # Build x-axis
    # -------------------------
    iters = wide["iter"].to_numpy()
    if args.xaxis == "iter":
        x_vals = list(iters)
        xlabel = "iter"
    else:
        x_vals, xlabel = convert_xaxis(iters, args.xaxis, control_file=args.control)
        x_vals = list(x_vals)

    # Series per molecule
    series = {m: wide[m].tolist() for m in molecules if m in wide.columns}

    # -------------------------
    # Export
    # -------------------------
    if args.export:
        workflow_name = getattr(args, "kind", "molfra")
        out = resolve_output_path(args.export, workflow_name)

        xcol = (
            xlabel.strip()
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        out_df = wide.copy()
        if args.xaxis != "iter":
            out_df.insert(0, xcol, x_vals)
        out_df.to_csv(out, index=False)
        print(f"[Done] Exported data to {out}")

    # -------------------------
    # Plot / save
    # -------------------------
    if args.save:
        workflow_name = getattr(args, "kind", "molfra")
        save_path = resolve_output_path(args.save, workflow_name)
    else:
        save_path = None

    if not args.title:
        if args.threshold is not None and not args.molecules:
            title = f"Species with max occurrence ≥ {args.threshold}"
        elif len(molecules) == 1:
            title = f"{molecules[0]} occurrence"
        else:
            title = "Molecule occurrence"
    else:
        title = args.title

    if (do_plot or save_path) and series:
        series_for_plot = [
            {"x": x_vals, "y": yvals, "label": name}
            for name, yvals in series.items()
        ]
        single_plot(
            series=series_for_plot,
            title=title,
            xlabel=xlabel,
            ylabel="occurrence",
            save=str(save_path) if save_path else None,
            legend=True,
        )
    elif not args.save and not args.export and not do_plot:
        print("ℹ️ No action selected. Use one or more of --plot, --save, --export.")

    return 0


# ============================================================
# Totals task
# ============================================================
def _molfra_total_task(args: argparse.Namespace) -> int:
    """
    Handle `reaxkit molfra total`:

    - If --plot and/or --save: create a multi-subplot figure of *all* totals
      (total_molecules, total_atoms, total_molecular_mass) vs chosen x-axis.
    - If --export: export all totals as CSV using get_molfra_totals_vs_axis().
    """
    # Normalize x-axis alias
    args.xaxis = normalize_choice(args.xaxis, domain="xaxis") or "iter"

    handler = MolFraHandler(args.file)
    handler._parse()

    # Ensure totals table exists so we can inspect which columns are present
    if not hasattr(handler, "_df_totals"):
        print("⚠️ Totals dataframe not found in MolFraHandler. Parse handler with an updated version first.")
        return 0

    df_raw = handler._df_totals
    if df_raw.empty:
        print("⚠️ Totals dataframe is empty.")
        return 0

    # Determine which totals columns exist
    totals_all = ("total_molecules", "total_atoms", "total_molecular_mass")
    quantities = [c for c in totals_all if c in df_raw.columns]
    if not quantities:
        print("⚠️ No totals columns (total_molecules / total_atoms / total_molecular_mass) found.")
        return 0

    # Get totals vs requested x-axis using analyzer helper
    df_tot = get_molfra_totals_vs_axis(
        handler,
        xaxis=args.xaxis,
        control_file=args.control,
        quantities=quantities,
    )
    if df_tot.empty:
        print("⚠️ Totals table is empty after axis conversion.")
        return 0

    # Apply frame selection (position-based after filtering)
    frames_sel = parse_frames(args.frames)
    df_tot = select_frames(df_tot, frames_sel)
    if df_tot.empty:
        print("⚠️ Totals table empty after frame selection.")
        return 0

    # Identify x-axis column (whatever is not a totals column)
    non_q = [c for c in df_tot.columns if c not in quantities]
    if not non_q:
        print("⚠️ Could not determine x-axis column.")
        return 0
    if "iter" in non_q and len(non_q) > 1:
        # Prefer non-iter if both iter & time-like column exist
        xcol = [c for c in non_q if c != "iter"][0]
    else:
        xcol = non_q[0]

    x_vals = df_tot[xcol].tolist()
    xlabel = xcol.replace("_", " ")

    # Flags: plotting vs exporting
    do_plot_or_save = bool(args.plot or args.save)
    workflow_name = getattr(args, "kind", "molfra")

    # ---------- Export all totals ----------
    if args.export:
        out_csv = resolve_output_path(args.export, workflow_name)
        df_tot.to_csv(out_csv, index=False)
        print(f"[Done] Exported totals to {out_csv}")

    # ---------- Multi-subplots for totals ----------
    if do_plot_or_save:
        # Build subplot series: one subplot per quantity
        subplots = []
        for q in quantities:
            subplots.append(
                [
                    {
                        "x": x_vals,
                        "y": df_tot[q].tolist(),
                        "label": q,
                    }
                ]
            )

        # Resolve save path (None → interactive only)
        save_path = resolve_output_path(args.save, workflow_name) if args.save else None

        title = args.title or f"Totals vs {xlabel}"
        multi_subplots(
            subplots=subplots,
            title=title,
            xlabel=xlabel,
            ylabel=["count", "count", "mass (a.u.)"],
            sharex=True,
            sharey=False,
            legend=True,
            figsize=(8.0, 2.5 * len(quantities)),
            save=save_path,
        )

    if not (args.plot or args.save or args.export):
        print("ℹ️ No action selected. Use one or more of --plot, --save, --export.")

    return 0


# ============================================================
# Largest-molecule task
# ============================================================
def _molfra_largest_task(args: argparse.Namespace) -> int:
    # Normalize x-axis alias
    args.xaxis = normalize_choice(args.xaxis, domain="xaxis") or "iter"

    handler = MolFraHandler(args.file)
    handler._parse()

    do_plot = bool(args.plot) and not bool(args.export)
    frames_sel = parse_frames(args.frames)

    # Decide mode: mass vs atoms
    mode_atoms = bool(args.atoms)
    mode_mass = bool(args.mass)

    if not mode_atoms and not mode_mass:
        # default to atoms if nothing specified
        mode_atoms = True

    workflow_name = getattr(args, "kind", "molfra")

    if mode_mass:
        # ===============================
        # Largest molecule mass vs x-axis
        # ===============================
        df = largest_molecule_by_individual_mass(handler)
        if df.empty:
            print("⚠️ No data for largest molecule by mass.")
            return 0

        df = select_frames(df, frames_sel)
        if df.empty:
            print("⚠️ No data for largest molecule after frame selection.")
            return 0

        iters = df["iter"].to_numpy()
        if args.xaxis == "iter":
            x_vals = list(iters)
            xlabel = "iter"
        else:
            x_vals, xlabel = convert_xaxis(iters, args.xaxis, control_file=args.control)
            x_vals = list(x_vals)

        y = df["molecular_mass"].to_list()

        # Export
        if args.export:
            out = resolve_output_path(args.export, workflow_name)
            xcol = (
                xlabel.strip()
                .lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
            )
            out_df = df.copy()
            if args.xaxis != "iter":
                out_df.insert(0, xcol, x_vals)
            out_df.to_csv(out, index=False)
            print(f"[Done] Exported data to {out}")

        # Save path
        if args.save:
            save_path = resolve_output_path(args.save, workflow_name)
        else:
            save_path = None

        if not args.title:
            title = "Largest molecule mass vs x-axis"
        else:
            title = args.title

        if do_plot or save_path:
            series_for_plot = [
                {
                    "x": x_vals,
                    "y": y,
                    "label": "largest_molecule_mass",
                }
            ]
            single_plot(
                series=series_for_plot,
                title=title,
                xlabel=xlabel,
                ylabel="Molecular mass",
                save=str(save_path) if save_path else None,
                legend=True,
            )
        elif not args.save and not args.export and not do_plot:
            print("ℹ️ No action selected. Use one or more of --plot, --save, --export.")

        return 0

    # ===============================
    # Largest-molecule atoms (wide)
    # ===============================
    df_wide = atoms_in_the_largest_molecule_wide_format(handler)
    if df_wide.empty:
        print("⚠️ No atom data available for largest molecule.")
        return 0

    df_wide = select_frames(df_wide, frames_sel)
    if df_wide.empty:
        print("⚠️ No atom data after frame selection.")
        return 0

    iters = df_wide["iter"].to_numpy()
    if args.xaxis == "iter":
        x_vals = list(iters)
        xlabel = "iter"
    else:
        x_vals, xlabel = convert_xaxis(iters, args.xaxis, control_file=args.control)
        x_vals = list(x_vals)

    element_cols = [c for c in df_wide.columns if c != "iter"]
    if not element_cols:
        print("⚠️ No element columns found to plot.")
        return 0

    series = {elem: df_wide[elem].tolist() for elem in element_cols}

    # Export
    if args.export:
        out = resolve_output_path(args.export, workflow_name)
        xcol = (
            xlabel.strip()
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        out_df = df_wide.copy()
        if args.xaxis != "iter":
            out_df.insert(0, xcol, x_vals)
        out_df.to_csv(out, index=False)
        print(f"[Done] Exported data to {out}")

    # Save path
    if args.save:
        save_path = resolve_output_path(args.save, workflow_name)
    else:
        save_path = None

    if not args.title:
        title = "Atom counts in largest molecule"
    else:
        title = args.title

    if (do_plot or save_path) and series:
        series_for_plot = [
            {"x": x_vals, "y": df_wide[elem].tolist(), "label": elem}
            for elem in element_cols
        ]
        single_plot(
            series=series_for_plot,
            title=title,
            xlabel=xlabel,
            ylabel="Atom count",
            save=str(save_path) if save_path else None,
            legend=True,
        )
    elif not args.save and not args.export and not do_plot:
        print("ℹ️ No action selected. Use one or more of --plot, --save, --export.")

    return 0


# ============================================================
# Registry
# ============================================================
def _add_common_molfra_axes_args(p: argparse.ArgumentParser) -> None:
    """Common axis and frame-selection flags for molfra-based tasks."""
    p.add_argument(
        "--xaxis",
        default="iter",
        help="X-axis mode. Canonical: 'iter', 'time', 'frame'.",
    )
    p.add_argument(
        "--control",
        default="control",
        help="Path to control file for time conversion (used when --xaxis time).",
    )
    p.add_argument(
        "--frames",
        default=None,
        help="Frame selection (position-based after filtering): 'start:stop[:step]' or 'i,j,k'.",
    )


def _add_common_molfra_output_args(
    p: argparse.ArgumentParser,
    *,
    plot_help: str = "Show the plot interactively.",
    save_help: str = "Save the plot (path or directory, resolved via resolve_output_path).",
    export_help: str = "Export the data table to CSV (path or directory, resolved via resolve_output_path).",
) -> None:
    """Common output / I/O flags for molfra-based tasks."""
    p.add_argument("--title", default=None, help="Custom plot title.")
    p.add_argument("--plot", action="store_true", help=plot_help)
    p.add_argument("--save", default=None, help=save_help)
    p.add_argument("--export", default=None, help=export_help)


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    # -----------------------
    # Occurrences subcommand
    # -----------------------
    p_occur = subparsers.add_parser(
        "occur",
        help="Get molecule occurrences across frames and optionally plot, save, or export.",
        description=(
            "Examples:\n"
            "  reaxkit molfra occur --molecules H2O N128Al128 --save water_and_slab_occurrence.png\n"
            "  reaxkit molfra occur --threshold 3 --exclude Pt --export species_with_max_occur.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p_occur.add_argument("--file", default="molfra.out", help="Path to molfra.out")

    group = p_occur.add_mutually_exclusive_group(required=False)
    group.add_argument("--molecules", nargs="+", default=None,
        help="One or more molecule types to include (e.g., H2O OH N128Al128).",
    )
    group.add_argument("--threshold", type=int, default=None,
        help="Auto-include all species whose max occurrence ≥ threshold.",
    )
    p_occur.add_argument("--exclude", nargs="*", default=None,
        help="Species to exclude when using --threshold (e.g., Pt).",
    )

    _add_common_molfra_axes_args(p_occur)
    _add_common_molfra_output_args(p_occur,
        plot_help="Show the plot interactively.",
        save_help="Save the plot (path or directory, resolved via resolve_output_path).",
        export_help="Export the data table to CSV (path or directory, resolved via resolve_output_path).",
    )

    p_occur.set_defaults(_run=_molfra_occur_task, kind="molfra")

    # -----------------------
    # Totals subcommand
    # -----------------------
    p_total = subparsers.add_parser(
        "total",
        help="Plot/export totals (molecules, atoms, mass) vs x-axis.",
        description=(
            "Examples:\n"
            "  reaxkit molfra total --file molfra_ig.out "
            "--export totals_data.csv --save totals_data.png\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p_total.add_argument("--file", default="molfra.out", help="Path to molfra.out")

    _add_common_molfra_axes_args(p_total)
    _add_common_molfra_output_args(p_total,
        plot_help="Show the multi-subplot figure interactively.",
        save_help="Save the multi-subplot figure (path or directory, resolved via resolve_output_path).",
        export_help="Export all totals to CSV (path or directory, resolved via resolve_output_path).",
    )

    p_total.set_defaults(_run=_molfra_total_task, kind="molfra")

    # -----------------------
    # Largest subcommand
    # -----------------------
    p_largest = subparsers.add_parser(
        "largest",
        help="Analyze the largest molecule (by individual mass or atom composition).",
        description=(
            "Examples:\n"
            "  reaxkit molfra largest --atoms --frames '0:30:2' "
            "--xaxis time --save largest.png --export largest.csv\n"
            "  reaxkit molfra largest --mass --xaxis time --export largest_mass.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p_largest.add_argument("--file", default="molfra.out", help="Path to molfra.out")

    mode_group = p_largest.add_mutually_exclusive_group(required=False)
    mode_group.add_argument("--atoms", dest="atoms", action="store_true",
        help="Use per-element atom counts for the largest molecule per iter (default).",
    )
    mode_group.add_argument("--mass", action="store_true",
        help="Use largest molecule individual mass vs x-axis.",
    )

    _add_common_molfra_axes_args(p_largest)
    _add_common_molfra_output_args(p_largest,
        plot_help="Show the plot interactively.",
        save_help="Save the plot (path or directory, resolved via resolve_output_path).",
        export_help="Export the data table to CSV (path or directory, resolved via resolve_output_path).",
    )

    p_largest.set_defaults(_run=_molfra_largest_task, kind="molfra")
