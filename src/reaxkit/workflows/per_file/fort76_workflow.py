"""
fort.76 restraint-analysis workflow for ReaxKit.

This workflow provides tools for reading, visualizing, and exporting data from
ReaxFF `fort.76` files, which record restraint targets and actual values during
MD or minimization runs.

It supports:
- Extracting and plotting a single restraint-related column versus iteration,
  frame index, or physical time.
- Comparing restraint target and actual values for a selected restraint index
  as a function of iteration, frame, or time.
- Converting the x-axis between iteration, frame, and time using the associated
  control file.
- Saving plots to disk or exporting data to CSV using standardized output paths.

The workflow is intended for diagnosing restraint behavior, convergence, and
stability in constrained ReaxFF simulations.
"""


from __future__ import annotations

import argparse

import pandas as pd

from reaxkit.io.handlers.fort76_handler import Fort76Handler
from reaxkit.analysis.per_file.fort76_analyzer import get_fort76_data, get_fort76_restraint_pairs
from reaxkit.utils.media.convert import convert_xaxis
from reaxkit.utils.media.plotter import single_plot
from reaxkit.utils.path import resolve_output_path


def _fort76_get_task(args: argparse.Namespace) -> int:
    """
    Handle: reaxkit fort76 get ...
    Plot/save/export ONE column vs iter/frame/time (derived from iter).
    """
    handler = Fort76Handler(args.file)
    handler._parse()

    df = get_fort76_data(handler, ["iter", args.ycol], dropna_rows=True).copy()

    xvals, xlabel = convert_xaxis(df["iter"].to_numpy(), args.xaxis, control_file=args.control)
    x = pd.Series(xvals)

    y_name = df.columns[1]
    y = df[y_name]

    workflow_name = args.kind  # same pattern as fort13_workflow :contentReference[oaicite:4]{index=4}

    # Export
    if args.export:
        out = resolve_output_path(args.export, workflow_name)
        pd.DataFrame({xlabel: x, y_name: y}).to_csv(out, index=False)
        print(f"[Done] Exported data to {out}.")

    # Plot / Save
    if args.plot:
        single_plot(
            x=x,
            y=y,
            title=args.title or f"{y_name} vs {args.xaxis}",
            xlabel=args.xlabel or xlabel,
            ylabel=args.ylabel or y_name,
            save=None,
        )
    elif args.save:
        out = resolve_output_path(args.save, workflow_name)
        single_plot(
            x=x,
            y=y,
            title=args.title or f"{y_name} vs {args.xaxis}",
            xlabel=args.xlabel or xlabel,
            ylabel=args.ylabel or y_name,
            save=out,
        )

    if not (args.plot or args.save or args.export):
        print("ℹ️ Nothing to do. Use --plot, --save <path>, --export <csv>.")
    return 0


def _fort76_respair_task(args: argparse.Namespace) -> int:
    """
    Handle: reaxkit fort76 respair ...
    Plot/save/export restraint target+actual vs iter/frame/time.
    """
    handler = Fort76Handler(args.file)
    handler._parse()

    df = get_fort76_restraint_pairs(handler, args.restraint, include_iter=True).copy()
    target_col = df.columns[1]
    actual_col = df.columns[2]

    xvals, xlabel = convert_xaxis(df["iter"].to_numpy(), args.xaxis, control_file=args.control)
    x = pd.Series(xvals)

    workflow_name = args.kind

    # Export
    if args.export:
        out = resolve_output_path(args.export, workflow_name)
        pd.DataFrame(
            {xlabel: x, target_col: df[target_col], actual_col: df[actual_col]}
        ).to_csv(out, index=False)
        print(f"Successfully exported data to {out}.")

    # Plot / Save
    if args.plot:
        single_plot(
            series=[
                {"x": x, "y": df[target_col], "label": target_col},
                {"x": x, "y": df[actual_col], "label": actual_col},
            ],
            title=args.title or f"Restraint {args.restraint}: target vs actual ({args.xaxis})",
            xlabel=args.xlabel or xlabel,
            ylabel=args.ylabel or "Value",
            legend=True,
            save=None,
        )
    elif args.save:
        out = resolve_output_path(args.save, workflow_name)
        single_plot(
            series=[
                {"x": x, "y": df[target_col], "label": target_col},
                {"x": x, "y": df[actual_col], "label": actual_col},
            ],
            title=args.title or f"Restraint {args.restraint}: target vs actual ({args.xaxis})",
            xlabel=args.xlabel or xlabel,
            ylabel=args.ylabel or "Value",
            legend=True,
            save=out,
        )

    if not (args.plot or args.save or args.export):
        print("ℹ️ Nothing to do. Use --plot, --save <path>, --export <csv>.")
    return 0


def _add_common_fort76_io_args(
    p: argparse.ArgumentParser,
    *,
    include_plot: bool = False,
) -> None:
    p.add_argument("--file", default="fort.76", help="Path to fort.76 file.")
    p.add_argument("--xaxis", default="iter", choices=["iter", "frame", "time"])
    p.add_argument("--control", default="control", help="Needed for --xaxis time.")
    if include_plot:
        p.add_argument("--plot", action="store_true", help="Show plot interactively.")
    p.add_argument("--save", default=None, help="Path to save plot image.")
    p.add_argument("--export", default=None, help="Path to export CSV data.")
    p.add_argument("--title", default=None)
    p.add_argument("--xlabel", default=None)
    p.add_argument("--ylabel", default=None)


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    # --- get ---
    p = subparsers.add_parser(
        "get",
        help="Plot, export, or save one fort.76 column vs iter/frame/time.\n",
        description=(
            "Examples:\n"
            "  reaxkit fort76 get --ycol E_res --xaxis time --save E_res_vs_time.png\n"
            "  reaxkit fort76 get --ycol r1_actual --xaxis frame --export r1_actual_vs_frame.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _add_common_fort76_io_args(p, include_plot=True)
    p.add_argument("--ycol", required=True, help="Column to plot/export (aliases allowed).")
    p.set_defaults(_run=_fort76_get_task)

    # --- respair ---
    p2 = subparsers.add_parser(
        "respair",
        help="Plot, export, or save restraint target+actual vs iter/frame/time.\n",
        description=(
            "Examples:\n"
            "  reaxkit fort76 respair --restraint 2 --xaxis time --control control --save r2_vs_time.png\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _add_common_fort76_io_args(p2, include_plot=True)
    p2.add_argument("--restraint", type=int, required=True, help="Restraint index (1-based).")
    p2.set_defaults(_run=_fort76_respair_task)

