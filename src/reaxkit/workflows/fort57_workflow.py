# reaxkit/workflows/fort57_workflow.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from reaxkit.io.fort57_handler import Fort57Handler
from reaxkit.analysis.fort57_analyzer import fort57_get
from reaxkit.utils.convert import convert_xaxis
from reaxkit.utils.plotter import single_plot
from reaxkit.utils.path import resolve_output_path


def _split_cols(s: str | None) -> List[str]:
    if not s:
        return []
    # support "a,b,c" or "a b c"
    toks: List[str] = []
    for part in s.replace(",", " ").split():
        p = part.strip()
        if p:
            toks.append(p)
    return toks


def get_task(args: argparse.Namespace) -> int:
    h = Fort57Handler(args.file)

    # --- decide which y columns user wants ---
    yreq = _split_cols(args.yaxis)
    if len(yreq) == 0:
        yreq = ["E_pot"]

    df_full = h.dataframe()

    if len(yreq) == 1 and yreq[0].lower() == "all":
        # all canonical columns except iter (we’ll add x separately)
        ycols = [c for c in ["E_pot", "T", "T_set", "RMSG", "nfc"] if c in df_full.columns or True]
    else:
        ycols = yreq

    # --- pull iter + requested y columns through analyzer (alias-aware) ---
    wanted = ["iter"] + ycols
    df = fort57_get(fort57_handler=h, cols=wanted, include_geo_descriptor=False)

    # --- convert x-axis using convert.py ---
    xvals, xlabel = convert_xaxis(df["iter"].to_numpy(), args.xaxis, control_file=args.control)

    # build output table: x + y
    xname = args.xaxis  # "iter" | "frame" | "time"
    out = pd.DataFrame({xname: xvals})
    for yc in ycols:
        if yc not in df.columns:
            raise KeyError(f"❌ Requested y column '{yc}' not found. Available: {', '.join(df.columns)}")
        out[yc] = df[yc].to_numpy()

    # --- export CSV (always x + y) ---
    if args.export:
        export_path = resolve_output_path(args.export, "fort57")
        out.to_csv(export_path, index=False)
        print(f"[Done] Exported CSV to {export_path}")

    # --- plot (one plot per y if multiple) ---
    if getattr(args, "plot", False) or args.save:
        base_save = args.save
        for yc in ycols:
            save_path = None
            if base_save:
                sp = Path(base_save)
                # suffix _<col> when multiple y columns
                if len(ycols) > 1:
                    sp = sp.with_name(f"{sp.stem}_{yc}{sp.suffix}")
                save_path = str(resolve_output_path(str(sp), "fort57"))

            single_plot(
                xvals,
                out[yc].to_numpy(),
                title=f"fort.57: {yc} vs {args.xaxis}",
                xlabel=xlabel,
                ylabel=yc,
                save=save_path,
            )

    # If user didn't plot or export, at least print a preview
    if not args.export and not getattr(args, "plot", False) and not args.save:
        print(out.head())

    return 0

###################################################################################

def _add_common_fort57_io_args(p: argparse.ArgumentParser, *, include_plot: bool = False) -> None:
    p.add_argument("--file", default="fort.57", help="Path to fort.57 file.")
    p.add_argument("--control", default="control", help="Path to control file (for --xaxis time).")
    p.add_argument("--xaxis", default="iter", choices=["iter", "frame", "time"],
                   help="X-axis: iter, frame, or time (time uses control:tstep).")
    p.add_argument("--yaxis", default="E_pot",
                   help="Y column(s): e.g. 'RMSG' or 'iter RMSG' or 'E_pot,T' or 'all'.")
    if include_plot:
        p.add_argument("--plot", action="store_true", help="Show plot interactively.")
    p.add_argument("--save", default=None, help="Path to save plot image (suffix _<col> if multiple y).")
    p.add_argument("--export", default=None, help="Path to export CSV (x + selected y columns).")

def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "get",
        help="Get/plot selected columns from fort.57 (with x-axis conversion).",
        description=(
            "Examples:\n"
            "  reaxkit fort57 get --yaxis RMSG --xaxis iter --save rmsg_vs_iter.png\n"
            "  reaxkit fort57 get --yaxis all --xaxis iter --export fort57_all.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    _add_common_fort57_io_args(p, include_plot=True)
    p.set_defaults(_run=get_task)
