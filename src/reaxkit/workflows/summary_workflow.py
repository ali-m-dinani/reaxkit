"""a workflow for getting summary.txt data"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Sequence, Union
import pandas as pd
from reaxkit.utils.units import UNITS
from reaxkit.io.summary_handler import SummaryHandler
from reaxkit.utils.alias import _resolve_alias, available_keys, normalize_choice
from reaxkit.analysis.plotter import single_plot
from reaxkit.utils.convert import convert_xaxis
from reaxkit.utils.frame_utils import parse_frames, select_frames

FramesT = Optional[Union[slice, Sequence[int]]]


def _summary_get_task(args: argparse.Namespace) -> int:
    handler = SummaryHandler(args.file)
    df = handler.dataframe().copy()

    # --- X axis: convert from 'iter' using convert_xaxis ---
    if "iter" not in df.columns:
        raise KeyError("Expected 'iter' column in parsed summary data.")
    xvals, xlabel = convert_xaxis(df["iter"].to_numpy(), args.xaxis)

    # --- Y axis: normalize alias -> canonical, then resolve ---
    y_canonical = normalize_choice(args.yaxis)          # e.g., "E_potential" -> "E_pot"
    try:
        ycol = _resolve_alias(handler, y_canonical)     # resolves to actual df column
    except KeyError as e:
        # helpful hint listing available keys
        raise KeyError(
            f"{e}\nTry one of: {available_keys(df.columns)}"
        )

    work = pd.DataFrame({"x": pd.Series(xvals, index=df.index), "y": df[ycol]})

    # --- Frame selection ---
    frames = parse_frames(args.frames)
    work = select_frames(work, frames)

    # --- Export CSV ---
    if args.export:
        out = Path(args.export)
        work.rename(columns={"x": xlabel, "y": ycol}).to_csv(out, index=False)

    # --- Save figure (no show) ---
    if args.save:
        single_plot(
            work["x"], work["y"],
            title=f"{ycol} vs {xlabel}",
            xlabel=xlabel, ylabel=f"{ycol} ({UNITS.get_sections_data(ycol, '') or ''})",
            save=args.save,
        )

    # --- Plot interactively ---
    if args.plot:
        single_plot(
            work["x"], work["y"],
            title=f"{ycol} vs {xlabel}",
            xlabel=xlabel, ylabel=f"{ycol} ({UNITS.get_sections_data(ycol, '') or ''})",
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
        help="Extract a column and optionally plot/save/export it.\n"
             "reaxkit summary get --yaxis E_pot --xaxis time --plot"
             "reaxkit summary get --file summary.txt --yaxis T --xaxis iter --frames 0:400:5 --save summary_T_vs_iter.png --export summary_T_vs_iter.csv"
    )
    _wire_get_flags(p)
