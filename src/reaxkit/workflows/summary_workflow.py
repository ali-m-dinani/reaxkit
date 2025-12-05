"""a workflow for processing summary.txt data"""

from __future__ import annotations
import argparse
from typing import Optional, Sequence, Union
import pandas as pd
from reaxkit.utils.units import UNITS
from reaxkit.utils.plotter import single_plot
from reaxkit.utils.convert import convert_xaxis
from reaxkit.utils.frame_utils import parse_frames, select_frames
from reaxkit.utils.path import resolve_output_path
from reaxkit.io.summary_handler import SummaryHandler
from reaxkit.utils.alias import available_keys
from reaxkit.analysis.summary_analyzer import get_summary

FramesT = Optional[Union[slice, Sequence[int]]]


def _summary_get_task(args: argparse.Namespace) -> int:
    handler = SummaryHandler(args.file)
    df = handler.dataframe().copy()

    # --- X axis: convert from 'iter' using convert_xaxis ---
    if "iter" not in df.columns:
        raise KeyError("Expected 'iter' column in parsed summary data.")
    xvals, xlabel = convert_xaxis(df["iter"].to_numpy(), args.xaxis)

    # --- Y axis: use analyzer-level helper for alias resolution ---
    try:
        y_series = get_summary(handler, args.yaxis)  # handles aliases + fallbacks
    except KeyError as e:
        # (optional) just re-raise, message already includes available keys
        # from summary_analyzer.get_summary
        raise e

    # Name of the resolved column (canonical or actual df column)
    ycol = y_series.name or args.yaxis

    # Build working DataFrame with aligned index
    work = pd.DataFrame(
        {
            "x": pd.Series(xvals, index=df.index),
            "y": y_series,
        }
    )

    # --- Frame selection ---
    frames = parse_frames(args.frames)
    work = select_frames(work, frames)

    workflow_name = args.kind

    # --- Export CSV ---
    if args.export:
        out = resolve_output_path(args.export, workflow_name)
        work.rename(columns={"x": xlabel, "y": ycol}).to_csv(out, index=False)
        print(f'[Done] successfully saved the data in {out}')

    # --- Save figure (no show) ---
    if args.save:
        out = resolve_output_path(args.save, workflow_name)
        single_plot(
            work["x"],
            work["y"],
            title=f"{ycol} vs {xlabel}",
            xlabel=xlabel,
            ylabel=f"{ycol} ({UNITS.get(ycol, '') or ''})",
            save=out,
        )

    # --- Plot interactively ---
    if args.plot:
        single_plot(
            work["x"],
            work["y"],
            title=f"{ycol} vs {xlabel}",
            xlabel=xlabel,
            ylabel=f"{ycol} ({UNITS.get(ycol, '') or ''})",
            save=None,
        )

    # --- No action fallback ---
    if not args.plot and not args.save and not args.export:
        print("ℹ️ No action selected. Use one or more of --plot, --save, --export.")
        print("Available keys:", ", ".join(available_keys(df.columns)))

    return 0


def _wire_get_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--file", default="summary.txt", help="Path to summary file")
    p.add_argument(
        "--xaxis",
        default="time",
        choices=["time", "iter", "frame"],
        help="X-axis domain (default: time)",
    )
    p.add_argument(
        "--yaxis",
        required=True,
        help="Y-axis feature/column (aliases allowed, e.g., 'E_potential' → 'E_pot')",
    )
    p.add_argument(
        "--frames",
        default=None,
        help="Frames to select: 'start:stop[:step]' or 'i,j,k' (default: all)",
    )
    p.add_argument("--plot", action="store_true", help="Show the plot interactively.")
    p.add_argument(
        "--save",
        default=None,
        help="Save the plot to a file (without showing). Provide a path.",
    )
    p.add_argument(
        "--export",
        default=None,
        help="Export the data to CSV. Provide a path.",
    )
    p.set_defaults(_run=_summary_get_task)


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    """
    Register 'summary' tasks. get can be used for example to plot potential energy vs time (auto-scaled fs/ps/ns).
    """
    p = subparsers.add_parser(
        "get",
        help="Extract a column and optionally plot/save/export it.",
        description=(
            "Examples:\n"
            "  reaxkit summary get --yaxis E_pot --xaxis time --plot\n"
            "  reaxkit summary get --file summary.txt --yaxis T --xaxis iter "
            "--frames 0:400:5 --save summary_T_vs_iter.png --export summary_T_vs_iter.csv"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _wire_get_flags(p)



