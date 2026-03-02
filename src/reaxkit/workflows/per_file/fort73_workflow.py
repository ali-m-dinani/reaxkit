"""
Energy-output workflow for ReaxKit (fort.73 / energylog / fort.58).

This workflow provides tools for reading, visualizing, and exporting energy
and thermodynamic data produced by ReaxFF simulations, as stored in
`fort.73`, `energylog`, or `fort.58` files.

It supports:
- Extracting individual or all available energy terms (e.g. Ebond, Eangle,
  Etot) as functions of iteration, frame index, or physical time.
- Converting the x-axis between iteration, frame, and time using the
  associated control file.
- Plotting selected energy components or saving them as image files.
- Exporting energy data to CSV for downstream analysis or comparison
  across simulation runs.

The workflow is designed for rapid inspection of ReaxFF energy evolution,
stability, and convergence behavior.
"""


from __future__ import annotations

import argparse, os
import pandas as pd

from reaxkit.analysis.timeseries import PartialEnergySeriesRequest, PartialEnergySeriesTask
from reaxkit.domain.data_models import PartialEnergyData
from reaxkit.engine.reaxff.adapter import ReaxFFAdapter
from reaxkit.presentation.convert import convert_xaxis
from reaxkit.presentation.plot import single_plot
from reaxkit.cli.path import resolve_output_path

def _fort73_get_task(args: argparse.Namespace) -> int:
    """
    Fort73 get task.

    Works on
    --------
    CLI workflow task arguments and helper utilities

    Parameters
    ----------
    args : argparse.Namespace
        Parameter description.

    Returns
    -------
    int
        Return value description.

    Examples
    --------
    >>>
    """
    DEFAULT_FILES = {
        "fort73": "fort.73",
        "energylog": "energylog",
        "fort58": "fort.58",
    }

    default_file = DEFAULT_FILES.get(args.kind, "fort.73")

    file_path = args.file or default_file
    requested_components = None if args.yaxis.lower() == "all" else [args.yaxis]
    series_result = PartialEnergySeriesTask().run(
        ReaxFFAdapter().load(
            PartialEnergyData,
            {
                "fort73": file_path,
                "input": file_path,
            },
        ),
        PartialEnergySeriesRequest(components=requested_components),
    )
    df = series_result.table
    if df is None or df.empty:
        raise ValueError("No partial-energy data found in fort.73 DataFrame.")

    # --- X-axis conversion ---
    iters = pd.to_numeric(df["iter"], errors="coerce").to_numpy(dtype=int)
    x_vals, x_label = convert_xaxis(iters, args.xaxis, control_file=args.control)

    # --- Y columns selection ---
    y_cols = list(dict.fromkeys(df["component"].astype(str).tolist()))
    if not y_cols:
        raise ValueError("No energy columns found in fort.73 DataFrame.")

    workflow_name = args.kind

    # --- Export CSV (optional, data only) ---
    if args.export:
        out = resolve_output_path(args.export, workflow_name)
        export_df = pd.DataFrame({"x": x_vals})
        for col in y_cols:
            export_df[col] = pd.to_numeric(
                df.loc[df["component"] == col, "value"],
                errors="coerce",
            ).to_numpy(dtype=float)
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
            y = pd.to_numeric(
                df.loc[df["component"] == col, "value"],
                errors="coerce",
            ).to_numpy(dtype=float)

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
    """
    Register workflow tasks under the given argparse subparser collection.

    Works on
    --------
    CLI workflow task arguments and helper utilities

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Parameter description.

    Examples
    --------
    >>>
    """
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
    p_get.add_argument("--file", default=None, help="Path to fort.73 / energylog file")
    p_get.add_argument("--yaxis", required=True, help="Energy column (e.g. Ebond) or 'all'")
    p_get.add_argument("--xaxis", default="iter", choices=["iter", "frame", "time"], help="X-axis type")
    p_get.add_argument("--control", default="control", help="Control file (used when xaxis=time)")
    p_get.add_argument("--export", default=None, help="Path to export CSV (x + selected y columns)")
    p_get.add_argument("--save", default=None, help="Path to save plot image (suffix _<col> if yaxis=all)")
    p_get.add_argument("--plot", action="store_true", help="If set, generate plot(s).")
    p_get.set_defaults(_run=_fort73_get_task)
