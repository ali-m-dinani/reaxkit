"""a workflow for getting fort.73 data"""

from __future__ import annotations

import argparse, os
import pandas as pd

from reaxkit.io.fort73_handler import Fort73Handler
from reaxkit.analysis.fort73_analyzer import get_fort73_data
from reaxkit.utils.convert import convert_xaxis
from reaxkit.utils.plotter import single_plot
from reaxkit.utils.path import resolve_output_path

def fort73_get_task(args: argparse.Namespace) -> int:
    handler = Fort73Handler(args.file)
    df = get_fort73_data(handler)

    if "iter" not in df.columns:
        raise KeyError("Column 'iter' not found in fort.73 DataFrame.")

    # --- X-axis conversion ---
    iters = df["iter"].to_numpy()
    x_vals, x_label = convert_xaxis(iters, args.xaxis, control_file=args.control)

    # --- Y columns selection ---
    if args.yaxis.lower() == "all":
        y_cols = [c for c in df.columns if c != "iter"]
        if not y_cols:
            raise ValueError("No energy columns found in fort.73 DataFrame.")
    else:
        if args.yaxis not in df.columns:
            raise KeyError(f"Requested y-axis column '{args.yaxis}' not found in fort.73 DataFrame.")
        y_cols = [args.yaxis]

    workflow_name = args.kind

    # --- Export CSV (optional, data only) ---
    if args.export:
        out = resolve_output_path(args.export, workflow_name)
        export_df = pd.DataFrame({"x": x_vals})
        for col in y_cols:
            export_df[col] = df[col].values
        export_df.to_csv(out, index=False)
        print(f"[Done] Exported fort.73 data to {out}")

    # --- Plot / Save (independent flags) ---
    if args.plot or args.save:
        save_base = None
        base_is_dir = False

        if args.save:
            # Resolve to reaxkit_out/... using same helper as export
            save_base = resolve_output_path(args.save, workflow_name)

            # Decide if save_base acts like a directory or a file
            if len(y_cols) > 1:
                # Multiple columns → treat as directory unless clearly a file name
                base_is_dir = os.path.isdir(save_base) or "." not in os.path.basename(save_base)
            else:
                base_is_dir = os.path.isdir(save_base)

            if base_is_dir:
                os.makedirs(save_base, exist_ok=True)
            else:
                parent = os.path.dirname(save_base)
                if parent:
                    os.makedirs(parent, exist_ok=True)

        for col in y_cols:
            y = df[col].values

            # Decide per-column save path (or None)
            save_path = None
            if save_base:
                if len(y_cols) == 1:
                    # Single column: allow explicit filename or directory
                    if base_is_dir:
                        save_path = os.path.join(save_base, f"{col}.png")
                    else:
                        save_path = save_base
                else:
                    # Multiple columns: directory or filename with suffix
                    if base_is_dir:
                        save_path = os.path.join(save_base, f"{col}.png")
                    else:
                        root, ext = os.path.splitext(save_base)
                        if not ext:
                            ext = ".png"
                        save_path = f"{root}_{col}{ext}"

            # single_plot handles both “show only” (save=None) and “save to file”
            single_plot(
                x_vals,
                y,
                title=f"{col} vs {x_label}",
                xlabel=x_label,
                ylabel=f"{col} (kcal/mole)",
                save=save_path,
            )

    return 0


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    p_get = subparsers.add_parser(
        "get",
        help="Extract and plot energy data from fort.73\n",
        description=(
            "Examples:\n"
            "  reaxkit fort73 get --yaxis Ebond --xaxis time --plot \n"
            "  reaxkit fort73 get --yaxis all --xaxis time --save reaxkit_outputs/fort73/ \n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p_get.add_argument("--file", default="fort.73", help="Path to fort.73 file")
    p_get.add_argument("--yaxis", required=True, help="Energy column (e.g. Ebond) or 'all'")
    p_get.add_argument("--xaxis", default="iter", choices=["iter", "frame", "time"], help="X-axis type")
    p_get.add_argument("--control", default="control", help="Control file (used when xaxis=time)")
    p_get.add_argument("--export", default=None, help="Path to export CSV (x + selected y columns)")
    p_get.add_argument("--save", default=None, help="Path to save plot image (suffix _<col> if yaxis=all)")
    p_get.add_argument("--plot", action="store_true", help="If set, generate plot(s).")
    p_get.set_defaults(_run=fort73_get_task)
