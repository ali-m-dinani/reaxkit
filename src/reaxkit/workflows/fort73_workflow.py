# reaxkit/workflows/fort73_workflow.py
from __future__ import annotations

import argparse, os
import pandas as pd

from reaxkit.io.fort73_handler import Fort73Handler
from reaxkit.analysis.fort73_analyzer import fort73_get
from reaxkit.utils.convert import convert_xaxis
from reaxkit.analysis.plotter import single_plot


def fort73_get_task(args: argparse.Namespace) -> int:
    handler = Fort73Handler(args.file)
    df = fort73_get(handler)

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

    # --- Export (optional) ---
    if args.export:
        export_df = pd.DataFrame({"x": x_vals})
        for col in y_cols: export_df[col] = df[col].values
        export_df.to_csv(args.export, index=False)
        print(f"[Done] Exported fort.73 data to {args.export}")

    # --- Plot (optional) ---
    if args.plot:
        # If saving with --yaxis all and save is a folder → ensure directory exists
        base_save = args.save
        if len(y_cols) > 1 and base_save:
            os.makedirs(base_save, exist_ok=True)

        if len(y_cols) == 1:
            col = y_cols[0]
            save_path = None
            if args.save:
                # If save is a directory → create file inside with column name
                if os.path.isdir(args.save):
                    save_path = os.path.join(args.save, f"{col}.png")
                else:
                    save_path = args.save  # Save directly to filename
            single_plot(
                x_vals,
                df[col].values,
                title=f"{col} vs {x_label}",
                xlabel=x_label,
                ylabel=col,
                save=save_path,
            )
            if save_path:
                print(f"[Done] Saved plot to {save_path}")
        else:
            # Multiple plots (yaxis=all) → save each column separately
            for col in y_cols:
                y = df[col].values
                save_path = None
                if args.save:
                    # Ensure it's a directory for multiple files
                    if os.path.isdir(args.save) or "." not in os.path.basename(args.save):
                        save_path = os.path.join(args.save, f"{col}.png")
                    else:
                        # If a file name is given, append suffix
                        root, ext = os.path.splitext(args.save)
                        save_path = f"{root}_{col}{ext}"

                single_plot(
                    x_vals,
                    y,
                    title=f"{col} vs {x_label}",
                    xlabel=x_label,
                    ylabel=col,
                    save=save_path,
                )
                if save_path:
                    print(f"[Done] Saved plot for {col} to {save_path}")

    return 0


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    # Here `subparsers` is the *task-level* subparsers for the already-created "fort73" parser.
    p_get = subparsers.add_parser("get", help="Extract and plot energy data from fort.73 || "
                                              "reaxkit fort73 get --yaxis Ebond --xaxis time --plot"
                                              "reaxkit fort73 get --yaxis all --xaxis time --plot --save fort73_processed")
    p_get.add_argument("--file", default="fort.73", help="Path to fort.73 file")
    p_get.add_argument("--yaxis", required=True, help="Energy column (e.g. Ebond) or 'all'")
    p_get.add_argument("--xaxis", default="iter", choices=["iter", "frame", "time"], help="X-axis type")
    p_get.add_argument("--control", default="control", help="Control file (used when xaxis=time)")
    p_get.add_argument("--export", default=None, help="Path to export CSV (x + selected y columns)")
    p_get.add_argument("--save", default=None, help="Path to save plot image (suffix _<col> if yaxis=all)")
    p_get.add_argument("--plot", action="store_true", help="If set, generate plot(s).")
    p_get.set_defaults(_run=fort73_get_task)
