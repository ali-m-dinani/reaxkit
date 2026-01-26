# src/reaxkit/workflows/fort76_workflow.py

"""
fort.76 workflow (modeled after fort13_workflow). :contentReference[oaicite:0]{index=0}

CLI:
  reaxkit fort76 get     --file fort.76 --ycol E_res --xaxis time --control control --save out.png --export out.csv
  reaxkit fort76 respair --file fort.76 --restraint 1 --xaxis iter --plot

- Uses resolve_output_path() for consistent output folders. :contentReference[oaicite:1]{index=1}
- Uses convert_xaxis() for iter→frame/time. :contentReference[oaicite:2]{index=2}
- Uses single_plot() for plotting/saving. :contentReference[oaicite:3]{index=3}
"""

from __future__ import annotations

import argparse

import pandas as pd

from reaxkit.io.fort76_handler import Fort76Handler
from reaxkit.analysis.fort76_analyzer import get_fort76_columns, get_restraint_pair
from reaxkit.utils.convert import convert_xaxis
from reaxkit.utils.plotter import single_plot
from reaxkit.utils.path import resolve_output_path


def fort76_get_task(args: argparse.Namespace) -> int:
    """
    Handle: reaxkit fort76 get ...
    Plot/save/export ONE column vs iter/frame/time (derived from iter).
    """
    handler = Fort76Handler(args.file)
    handler._parse()

    df = get_fort76_columns(handler, ["iter", args.ycol], dropna_rows=True).copy()

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


def fort76_respair_task(args: argparse.Namespace) -> int:
    """
    Handle: reaxkit fort76 respair ...
    Plot/save/export restraint target+actual vs iter/frame/time.
    """
    handler = Fort76Handler(args.file)
    handler._parse()

    df = get_restraint_pair(handler, args.restraint, include_iter=True).copy()
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
    p.set_defaults(_run=fort76_get_task)

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
    p2.set_defaults(_run=fort76_respair_task)

