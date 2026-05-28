"""General-purpose plotting workflow for ReaxKit.

This workflow provides flexible plotting utilities for arbitrary tabular data
(text, CSV, TSV, or whitespace-delimited files), without assuming any specific
ReaxFF file format or column headers.

It supports multiple plot types, including:
- single and multi-series line or scatter plots,
- directed (arrowed) line plots,
- dual y-axis plots,
- tornado plots for sensitivity-style visualization,
- 3D scatter plots with scalar coloring,
- 2D aggregated heatmaps projected from 3D data.

Columns are selected using simple 1-based column tokens (e.g. c1, c2, c3),
making the workflow suitable for rapid visualization of simulation outputs,
summaries, and post-processed analysis tables.

**Usage context**

- Command routing: Resolve CLI aliases and normalized command names.
- Task execution: Build request objects and invoke registered tasks.
- Output handling: Forward results to table, plot, export, or report flows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any, Sequence
import numpy as np
import pandas as pd

from reaxkit.cli.path import resolve_output_path
from reaxkit.presentation.plot import (
    single_plot,
    directed_plot,
    dual_yaxis_plot,
    tornado_plot,
    scatter3d_points,
    heatmap2d_from_3d,
)

ALL_COMMANDS = ("gen-plot",)
ALL_LEGACY_COMMANDS = ("gen_plot", "plotter")


# ---------- Helpers ----------

def _normalize_save_path(args: argparse.Namespace) -> None:
    """Normalize save path."""
    if not getattr(args, "save", None):
        return
    out = resolve_output_path(str(args.save), workflow="plotter")
    args.save = str(out)

def _load_table(path: str | Path) -> pd.DataFrame:
    """
    Load table.

    Works on
    -----
    CLI workflow task arguments and helper utilities

    Parameters
    -----
    path : str | Path
        Parameter description.

    Returns
    -----
    pd.DataFrame
        Return value description.

    Examples
    -----
    >>>
    """
    path = Path(path)
    suf = path.suffix.lower()

    # 1) CSV / TSV
    if suf == ".csv":
        return pd.read_csv(path, header=None, comment="#", engine="python")
    if suf in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t", header=None, comment="#", engine="python")

    # 2) Generic text: try whitespace first (summary.txt style)
    try:
        return pd.read_csv(
            path,
            sep=r"\s+",
            engine="python",
            header=None,
            comment="#",
            on_bad_lines="skip",
        )
    except Exception:
        # 3) Last resort: let pandas sniff delimiter
        return pd.read_csv(
            path,
            sep=None,
            engine="python",
            header=None,
            comment="#",
            on_bad_lines="skip",
        )



def _parse_col_token(token: str) -> int:
    """
    Parse a column selector like 'c1', 'c2', '3', etc. into a zero-based index.

    - 'c1' -> 0
    - 'c3' -> 2
    - '1'  -> 0
    - '3'  -> 2
    """
    t = token.strip().lower()
    if t.startswith("c"):
        t = t[1:]
    if not t:
        raise ValueError(f"Invalid column token: '{token}'")
    idx = int(t) - 1
    if idx < 0:
        raise ValueError(f"Column indices are 1-based: got '{token}'")
    return idx


def _parse_col_list(spec: str) -> List[int]:
    """
    Parse col list.

    Works on
    -----
    CLI workflow task arguments and helper utilities

    Parameters
    -----
    spec : str
        Parameter description.

    Returns
    -----
    List[int]
        Return value description.

    Examples
    -----
    >>>
    """
    tokens = [s for s in spec.split(",") if s.strip()]
    if not tokens:
        raise ValueError(f"Empty column specification: '{spec}'")
    return [_parse_col_token(tok) for tok in tokens]


def _extract_numeric_subframe(df_raw: pd.DataFrame, col_indices: Sequence[int]) -> pd.DataFrame:
    """
    From a raw DataFrame, take the subset of columns with the given indices and
    convert each to numeric, coercing invalid entries to NaN.

    Returns a new DataFrame whose columns are named 'c1', 'c2', ... in the sense
    of 1-based original indices, and which has rows where **any** of the selected
    columns is NaN dropped (so all selected columns are aligned).
    """
    cols_unique = sorted(set(col_indices))
    data: Dict[str, Any] = {}
    for idx in cols_unique:
        if idx >= df_raw.shape[1]:
            raise IndexError(
                f"Requested column c{idx + 1}, but input file has only "
                f"{df_raw.shape[1]} columns."
            )
        name = f"c{idx + 1}"
        data[name] = pd.to_numeric(df_raw.iloc[:, idx], errors="coerce")

    sub = pd.DataFrame(data)
    # Drop rows where any selected column is NaN (header lines, malformed rows, etc.)
    sub = sub.dropna(how="any").reset_index(drop=True)
    if sub.empty:
        raise ValueError("After cleaning, no valid numeric rows remain to plot.")
    return sub


# ---------- Tasks ----------

def _plotter_single_task(args: argparse.Namespace) -> int:
    """
    reaxkit plotter single --file summary.txt --xaxis c1,c3 --yaxis c2,c4 [--save out.png] [--scatter]

    - If len(xaxis) == len(yaxis): pair-wise (x_i, y_i) series.
    - If len(xaxis) == 1 and len(yaxis) >= 1: use the same x for all y columns.
    """
    _normalize_save_path(args)
    df_raw = _load_table(args.file)
    x_indices = _parse_col_list(args.xaxis)
    y_indices = _parse_col_list(args.yaxis)
    all_indices = x_indices + y_indices
    sub = _extract_numeric_subframe(df_raw, all_indices)

    # Build series list for single_plot
    series: List[Dict[str, Any]] = []

    def col_name(idx: int) -> str:
        """Col name.

        Execute the workflow function for this command path and return the
        computed result for downstream CLI handling.

        Parameters
        -----
        idx : Any
            Function argument.

        Returns
        -----
        str
            Function return value.

        Examples
        -----
        >>> # See workflow CLI usage for concrete examples.
        """
        return f"c{idx + 1}"

    x_names = [col_name(i) for i in x_indices]
    y_names = [col_name(i) for i in y_indices]

    # Series construction
    if len(x_indices) == len(y_indices):
        for xi, yi in zip(x_indices, y_indices):
            xn = col_name(xi)
            yn = col_name(yi)
            series.append({
                "x": sub[xn],
                "y": sub[yn],
                "label": f"{yn} vs {xn}",
            })
    elif len(x_indices) == 1 and len(y_indices) >= 1:
        xi = x_indices[0]
        xn = col_name(xi)
        x_series = sub[xn]
        for yi in y_indices:
            yn = col_name(yi)
            series.append({
                "x": x_series,
                "y": sub[yn],
                "label": f"{yn} vs {xn}",
            })
    else:
        raise ValueError(
            "For 'single' plot: either provide the same number of x and y columns "
            "OR one x column with multiple y columns."
        )

    plot_type = "scatter" if args.scatter else "line"
    title = args.title or "single_plot"

    if not args.plot and not args.save:
        print("Nothing to do. Use --plot to display or --save filename to export.")
        return 0

    if args.plot:
        single_plot(
            series=series,
            title=title,
            xlabel=args.xlabel or "x",
            ylabel=args.ylabel or "y",
            legend=True,
            plot_type=plot_type,
        )
        return 0
    elif args.save:
        single_plot(
            series=series,
            title=title,
            xlabel=args.xlabel or "x",
            ylabel=args.ylabel or "y",
            legend=True,
            save=args.save,
            plot_type=plot_type,
        )
        return 0


def _plotter_directed_task(args: argparse.Namespace) -> int:
    """
    Plotter directed task.

    Works on
    -----
    CLI workflow task arguments and helper utilities

    Parameters
    -----
    args : argparse.Namespace
        Parameter description.

    Returns
    -----
    int
        Return value description.

    Examples
    -----
    >>>
    """
    _normalize_save_path(args)
    df_raw = _load_table(args.file)
    x_idx = _parse_col_token(args.xaxis)
    y_idx = _parse_col_token(args.yaxis)
    sub = _extract_numeric_subframe(df_raw, [x_idx, y_idx])

    xn = f"c{x_idx + 1}"
    yn = f"c{y_idx + 1}"

    if not args.plot and not args.save:
        print("Nothing to do. Use --plot to display or --save filename to export.")
        return 0

    if args.plot:
        directed_plot(
            x=sub[xn].to_numpy(),
            y=sub[yn].to_numpy(),
            title=args.title or "directed_plot",
            xlabel=args.xlabel or xn,
            ylabel=args.ylabel or yn,
        )
        return 0
    elif args.save:
        directed_plot(
            x=sub[xn].to_numpy(),
            y=sub[yn].to_numpy(),
            title=args.title or "directed_plot",
            xlabel=args.xlabel or xn,
            ylabel=args.ylabel or yn,
            save=args.save,
        )
        return 0


def _plotter_dual_task(args: argparse.Namespace) -> int:
    """
    Plotter dual task.

    Works on
    -----
    CLI workflow task arguments and helper utilities

    Parameters
    -----
    args : argparse.Namespace
        Parameter description.

    Returns
    -----
    int
        Return value description.

    Examples
    -----
    >>>
    """
    _normalize_save_path(args)
    df_raw = _load_table(args.file)
    x_idx = _parse_col_token(args.xaxis)
    y1_idx = _parse_col_token(args.y1)
    y2_idx = _parse_col_token(args.y2)

    sub = _extract_numeric_subframe(df_raw, [x_idx, y1_idx, y2_idx])

    xn = f"c{x_idx + 1}"
    y1n = f"c{y1_idx + 1}"
    y2n = f"c{y2_idx + 1}"

    if not args.plot and not args.save:
        print("Nothing to do. Use --plot to display or --save filename to export.")
        return 0

    if args.plot:
        dual_yaxis_plot(
            x=sub[xn].to_numpy(),
            y1=sub[y1n].to_numpy(),
            y2=sub[y2n].to_numpy(),
            title=args.title or "dual_yaxis_plot",
            xlabel=args.xlabel or xn,
            ylabel1=args.ylabel1 or y1n,
            ylabel2=args.ylabel2 or y2n,
        )
        return 0
    elif args.save:
        dual_yaxis_plot(
            x=sub[xn].to_numpy(),
            y1=sub[y1n].to_numpy(),
            y2=sub[y2n].to_numpy(),
            title=args.title or "dual_yaxis_plot",
            xlabel=args.xlabel or xn,
            ylabel1=args.ylabel1 or y1n,
            ylabel2=args.ylabel2 or y2n,
            save=args.save,
        )
        return 0


def _plotter_tornado_task(args: argparse.Namespace) -> int:
    """
    reaxkit plotter tornado --file summary.txt --label c1 --min c2 --max c3 [--median c4]
                             [--top 10] [--vline 0.0] [--save out.png]
    """
    _normalize_save_path(args)
    df_raw = _load_table(args.file)

    label_idx = _parse_col_token(args.label)
    min_idx = _parse_col_token(args.min)
    max_idx = _parse_col_token(args.max)
    median_idx = _parse_col_token(args.median) if args.median else None

    # Labels stay as strings
    labels = df_raw.iloc[:, label_idx].astype(str)

    # Numeric columns
    min_series = pd.to_numeric(df_raw.iloc[:, min_idx], errors="coerce")
    max_series = pd.to_numeric(df_raw.iloc[:, max_idx], errors="coerce")
    median_series = (
        pd.to_numeric(df_raw.iloc[:, median_idx], errors="coerce")
        if median_idx is not None
        else None
    )

    data = pd.DataFrame({
        "label": labels,
        "min": min_series,
        "max": max_series,
    })
    if median_series is not None:
        data["median"] = median_series

    # Drop rows where min or max is NaN
    data = data.dropna(subset=["min", "max"]).reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid rows for tornado plot after cleaning.")

    median_vals = data["median"] if "median" in data.columns else None

    if not args.plot and not args.save:
        print("Nothing to do. Use --plot to display or --save filename to export.")
        return 0

    if args.plot:
        tornado_plot(
            labels=data["label"].tolist(),
            min_vals=data["min"].to_numpy(),
            max_vals=data["max"].to_numpy(),
            median_vals=median_vals.to_numpy() if median_vals is not None else None,
            title=args.title or "Tornado Plot",
            xlabel=args.xlabel or "Value",
            ylabel=args.ylabel or "Parameter",
            top=args.top or 0,
            vline=args.vline,
        )
        return 0
    elif args.save:
        tornado_plot(
            labels=data["label"].tolist(),
            min_vals=data["min"].to_numpy(),
            max_vals=data["max"].to_numpy(),
            median_vals=median_vals.to_numpy() if median_vals is not None else None,
            title=args.title or "Tornado Plot",
            xlabel=args.xlabel or "Value",
            ylabel=args.ylabel or "Parameter",
            top=args.top or 0,
            vline=args.vline,
            save=args.save,
        )
        return 0


def _plotter_scatter3d_task(args: argparse.Namespace) -> int:
    """
    Plotter scatter3d task.

    Works on
    -----
    CLI workflow task arguments and helper utilities

    Parameters
    -----
    args : argparse.Namespace
        Parameter description.

    Returns
    -----
    int
        Return value description.

    Examples
    -----
    >>>
    """
    _normalize_save_path(args)
    df_raw = _load_table(args.file)
    x_idx = _parse_col_token(args.x)
    y_idx = _parse_col_token(args.y)
    z_idx = _parse_col_token(args.z)
    v_idx = _parse_col_token(args.value)

    sub = _extract_numeric_subframe(df_raw, [x_idx, y_idx, z_idx, v_idx])

    xn = f"c{x_idx + 1}"
    yn = f"c{y_idx + 1}"
    zn = f"c{z_idx + 1}"
    vn = f"c{v_idx + 1}"

    coords = np.column_stack(
        [sub[xn].to_numpy(), sub[yn].to_numpy(), sub[zn].to_numpy()]
    )
    values = sub[vn].to_numpy()

    if not args.plot and not args.save:
        print("Nothing to do. Use --plot to display or --save filename to export.")
        return 0

    if args.plot:
        scatter3d_points(
            coords=coords,
            values=values,
            title=args.title or "scatter3d",
        )
        return 0
    elif args.save:
        scatter3d_points(
            coords=coords,
            values=values,
            title=args.title or "scatter3d",
            save=args.save,
        )
        return 0


def _plotter_heatmap2d_task(args: argparse.Namespace) -> int:
    """
    reaxkit plotter heatmap2d --file summary.txt --x c1 --y c2 --z c3 --value c4
                              [--plane xy|xz|yz] [--bins 50] [--save out.png]

    Uses heatmap2d_from_3d under the hood.
    """
    _normalize_save_path(args)
    df_raw = _load_table(args.file)
    x_idx = _parse_col_token(args.x)
    y_idx = _parse_col_token(args.y)
    z_idx = _parse_col_token(args.z)
    v_idx = _parse_col_token(args.value)

    sub = _extract_numeric_subframe(df_raw, [x_idx, y_idx, z_idx, v_idx])

    xn = f"c{x_idx + 1}"
    yn = f"c{y_idx + 1}"
    zn = f"c{z_idx + 1}"
    vn = f"c{v_idx + 1}"

    coords = np.column_stack(
        [sub[xn].to_numpy(), sub[yn].to_numpy(), sub[zn].to_numpy()]
    )
    values = sub[vn].to_numpy()

    bins = args.bins
    if "," in str(bins):
        # allow e.g. "--bins 50,100"
        bx, by = (int(p.strip()) for p in str(bins).split(","))
        bins_arg = (bx, by)
    else:
        bins_arg = int(bins)

    if not args.plot and not args.save:
        print("Nothing to do. Use --plot to display or --save filename to export.")
        return 0

    if args.plot:
        heatmap2d_from_3d(
            coords=coords,
            values=values,
            plane=args.plane,
            bins=bins_arg,
            title=args.title or "2D aggregated heatmap",
            save=args.save,
        )
        return 0
    elif args.save:
        heatmap2d_from_3d(
            coords=coords,
            values=values,
            plane=args.plane,
            bins=bins_arg,
            title=args.title or "2D aggregated heatmap",
            save=args.save,
        )
        return 0

# ---------- Registration ----------

def _add_common_io_args(p: argparse.ArgumentParser) -> None:
    """
    Add shared CLI arguments to the provided parser.

    Works on
    --------
    CLI workflow task arguments and helper utilities

    Parameters
    -----
    p : argparse.ArgumentParser
        Parameter description.

    Examples
    -----
    >>>
    """
    p.add_argument(
        "--file",
        required=True,
        help="Path to input txt/csv/tsv table. Example: --file table.csv, which loads plotting data from that file.",
    )
    p.add_argument(
        "--save",
        default=None,
        help="Path to save plot (file or directory). Example: --save figures/msd.png, which writes the generated figure to that path.",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Optional custom plot title. Example: --title \"MSD vs Time\", which overrides the default auto title.",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Generate and display/save the plot. Example: --plot, which opens the plot interactively when supported.",
    )


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Build parser.

    Execute the workflow function for this command path and return the
    computed result for downstream CLI handling.

    Parameters
    -----
    parser : Any
        Function argument.
    command : Any
        Function argument.

    Returns
    -----
    argparse.ArgumentParser
        Function return value.

    Examples
    -----
    >>> # See workflow CLI usage for concrete examples.
    """
    _ = command
    parser.set_defaults(command="gen-plot")
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "General-purpose plotting command for tabular data files.\n"
        "Select plot behavior with --type and provide the related flags for that type.\n\n"
        "Examples:\n"
        "  1. Single plot:\n"
        "   reaxkit gen-plot --type single --file msd.csv --xaxis c1 --yaxis c2 --plot\n\n"
        "  2. Directed plot:\n"
        "   reaxkit gen-plot --type directed --file table.csv --xaxis c1 --yaxis c2 --save directed.png\n\n"
        "  3. Heatmap2d plot:\n"
        "   reaxkit gen-plot --type heatmap2d --file table.csv --x c1 --y c2 --z c3 --value c4 --plane xz --bins 100,80 --save heat_xz.png"
    )
    parser.add_argument("--type", required=True, choices=["single", "directed", "dual", "tornado", "scatter3d", "heatmap2d"], help="Plot type selector. Example: --type single, which enables single-plot mode.")
    _add_common_io_args(parser)
    parser.add_argument("--xaxis", default=None, help="X column(s), format depends on --type. Example: --xaxis c1,c3, which selects x columns for single/dual/directed.")
    parser.add_argument("--yaxis", default=None, help="Y column(s), format depends on --type. Example: --yaxis c2,c4, which selects y columns for single/directed.")
    parser.add_argument("--xlabel", default=None, help="Optional x-axis label. Example: --xlabel Time, which customizes axis text.")
    parser.add_argument("--ylabel", default=None, help="Optional y-axis label. Example: --ylabel MSD, which customizes axis text.")
    parser.add_argument("--scatter", action="store_true", help="Use scatter instead of line (single mode). Example: --scatter, which switches marker-style rendering.")
    parser.add_argument("--y1", default=None, help="Left y-axis column for dual mode. Example: --y1 c2, which maps c2 to left axis.")
    parser.add_argument("--y2", default=None, help="Right y-axis column for dual mode. Example: --y2 c3, which maps c3 to right axis.")
    parser.add_argument("--ylabel1", default=None, help="Optional left y-axis label. Example: --ylabel1 Temp, which labels left axis.")
    parser.add_argument("--ylabel2", default=None, help="Optional right y-axis label. Example: --ylabel2 Pressure, which labels right axis.")
    parser.add_argument("--label", default=None, help="Label column for tornado mode. Example: --label c1, which provides tornado labels.")
    parser.add_argument("--min", dest="min", default=None, help="Minimum-value column for tornado mode. Example: --min c2, which sets low bound column.")
    parser.add_argument("--max", dest="max", default=None, help="Maximum-value column for tornado mode. Example: --max c3, which sets high bound column.")
    parser.add_argument("--median", default=None, help="Optional median column for tornado mode. Example: --median c4, which adds median markers.")
    parser.add_argument("--top", type=int, default=0, help="Top-N for tornado mode. Example: --top 10, which keeps widest 10 bars.")
    parser.add_argument("--vline", type=float, default=None, help="Reference vertical line for tornado mode. Example: --vline 0.0, which draws baseline.")
    parser.add_argument("--x", default=None, help="X coordinate column for scatter3d/heatmap2d. Example: --x c1, which maps column 1 to x.")
    parser.add_argument("--y", default=None, help="Y coordinate column for scatter3d/heatmap2d. Example: --y c2, which maps column 2 to y.")
    parser.add_argument("--z", default=None, help="Z coordinate column for scatter3d/heatmap2d. Example: --z c3, which maps column 3 to z.")
    parser.add_argument("--value", default=None, help="Value column for scatter3d/heatmap2d color/aggregation. Example: --value c4, which supplies scalar values.")
    parser.add_argument("--plane", choices=["xy", "xz", "yz"], default="xy", help="Projection plane for heatmap2d. Example: --plane xz, which projects onto XZ.")
    parser.add_argument("--bins", default="50", help="Heatmap bins: int or nx,ny. Example: --bins 100,80, which sets asymmetric grid resolution.")
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    """Run main.

    Execute the workflow function for this command path and return the
    computed result for downstream CLI handling.

    Parameters
    -----
    command : Any
        Function argument.
    args : Any
        Function argument.

    Returns
    -----
    int
        Function return value.

    Examples
    -----
    >>> # See workflow CLI usage for concrete examples.
    """
    _ = command
    dispatch = {
        "single": _plotter_single_task,
        "directed": _plotter_directed_task,
        "dual": _plotter_dual_task,
        "tornado": _plotter_tornado_task,
        "scatter3d": _plotter_scatter3d_task,
        "heatmap2d": _plotter_heatmap2d_task,
    }
    runner = dispatch.get(str(args.type))
    if runner is None:
        raise ValueError(f"Unsupported --type {args.type!r}")
    return runner(args)
