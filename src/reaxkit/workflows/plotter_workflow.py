"""Workflow for general purpose plotting working on any tabular data"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any, Sequence

import numpy as np
import pandas as pd

from reaxkit.utils.plotter import (
    single_plot,
    directed_plot,
    dual_yaxis_plot,
    tornado_plot,
    scatter3d_points,
    heatmap2d_from_3d,
)


# ---------- Helpers ----------

def _load_table(path: str | Path) -> pd.DataFrame:
    """
    Load a generic text table (csv/tsv/space-delimited) with no assumptions about headers.

    - Uses sep=None + engine="python" so pandas infers the delimiter.
    - Uses header=None so we don't lose the first data row regardless of header presence.
    - We keep the raw DataFrame (may include strings); numeric conversion is done
      per-column in the plotting tasks.
    """
    return pd.read_csv(
        path,
        sep=r"\s+",  # treat any run of whitespace as one separator
        engine="python",
        header=None,
        comment="#",
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
    Parse a comma-separated list like 'c1,c3' -> [0, 2].
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

def plotter_single_task(args: argparse.Namespace) -> int:
    """
    reaxkit plotter single --file summary.txt --xaxis c1,c3 --yaxis c2,c4 [--save out.png] [--scatter]

    - If len(xaxis) == len(yaxis): pair-wise (x_i, y_i) series.
    - If len(xaxis) == 1 and len(yaxis) >= 1: use the same x for all y columns.
    """
    df_raw = _load_table(args.file)
    x_indices = _parse_col_list(args.xaxis)
    y_indices = _parse_col_list(args.yaxis)
    all_indices = x_indices + y_indices
    sub = _extract_numeric_subframe(df_raw, all_indices)

    # Build series list for single_plot
    series: List[Dict[str, Any]] = []

    def col_name(idx: int) -> str:
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


def plotter_directed_task(args: argparse.Namespace) -> int:
    """
    reaxkit plotter directed --file summary.txt --xaxis c1 --yaxis c2 [--save dir.png]
    """
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


def plotter_dual_task(args: argparse.Namespace) -> int:
    """
    reaxkit plotter dual --file summary.txt --xaxis c1 --y1 c2 --y2 c3 [--save dual.png]
    """
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


def plotter_tornado_task(args: argparse.Namespace) -> int:
    """
    reaxkit plotter tornado --file summary.txt --label c1 --min c2 --max c3 [--median c4]
                             [--top 10] [--vline 0.0] [--save out.png]
    """
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


def plotter_scatter3d_task(args: argparse.Namespace) -> int:
    """
    reaxkit plotter scatter3d --file summary.txt --x c1 --y c2 --z c3 --value c4 [--save out.png]
    """
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


def plotter_heatmap2d_task(args: argparse.Namespace) -> int:
    """
    reaxkit plotter heatmap2d --file summary.txt --x c1 --y c2 --z c3 --value c4
                              [--plane xy|xz|yz] [--bins 50] [--save out.png]

    Uses heatmap2d_from_3d under the hood.
    """
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
    p.add_argument(
        "--file",
        required=True,
        help="Path to input txt/csv/tsv table.",
    )
    p.add_argument(
        "--save",
        default=None,
        help="Path to save plot (file or directory). If omitted, show interactively.",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Optional custom plot title.",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Generate and display/save the plot",
    )


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    """
    Register subcommands under the existing 'plotter' command:

      reaxkit plotter single   ...
      reaxkit plotter directed ...
      reaxkit plotter dual     ...
      reaxkit plotter tornado  ...
      reaxkit plotter scatter3d...
      reaxkit plotter heatmap2d...

    Save plots with titles and axis labels:
      reaxkit plotter single --file summary.txt --xaxis c1 --yaxis c2 \
      --title "Stress vs Strain" --xlabel "Strain" --ylabel "Stress" \
      --save stress_strain.png

    """

    # ---- single ----
    p_single = subparsers.add_parser(
        "single",
        help="Plot one or multiple y columns vs x columns (line or scatter). || "
             "reaxkit plotter single --file summary.txt --xaxis c1 --yaxis c2 --plot || "
             "reaxkit plotter single --file summary.txt --xaxis c1 --yaxis c2 --scatter --plot || "
             "reaxkit plotter single --file summary.txt --xaxis c1,c3 --yaxis c2,c4 --plot || "
             "reaxkit plotter single --file summary.txt --xaxis c1 --yaxis c2,c3,c4 --scatter --plot",
    )
    _add_common_io_args(p_single)
    p_single.add_argument("--xaxis", required=True,
        help="Comma-separated list of x columns (e.g., 'c1' or 'c1,c3').",
    )
    p_single.add_argument("--yaxis", required=True,
        help="Comma-separated list of y columns (e.g., 'c2' or 'c2,c4').",
    )
    p_single.add_argument("--xlabel", default=None, help="Optional x-axis label.")
    p_single.add_argument("--ylabel", default=None, help="Optional y-axis label.")
    p_single.add_argument("--scatter", action="store_true", help="Use scatter instead of line plot.")
    p_single.set_defaults(_run=plotter_single_task)

    # ---- directed ----
    p_directed = subparsers.add_parser(
        "directed",
        help="Line plot with arrows showing direction along the path. || "
             "reaxkit plotter directed --file summary.txt --xaxis c1 --yaxis c2 --save directed.png",
    )
    _add_common_io_args(p_directed)
    p_directed.add_argument("--xaxis", required=True, help="Single x column (e.g., 'c1').")
    p_directed.add_argument("--yaxis", required=True, help="Single y column (e.g., 'c2').")
    p_directed.add_argument("--xlabel", default=None, help="Optional x-axis label.")
    p_directed.add_argument("--ylabel", default=None, help="Optional y-axis label.")
    p_directed.set_defaults(_run=plotter_directed_task)

    # ---- dual ----
    p_dual = subparsers.add_parser(
        "dual",
        help="Dual y-axis plot: one x column, two y columns. || "
             "reaxkit plotter dual --file summary.txt --xaxis c1 --y1 c2 --y2 c3 --save dual_plot.png",
    )
    _add_common_io_args(p_dual)
    p_dual.add_argument("--xaxis", required=True, help="Single x column (e.g., 'c1').")
    p_dual.add_argument("--y1", required=True, help="Left y-axis column (e.g., 'c2').")
    p_dual.add_argument("--y2", required=True, help="Right y-axis column (e.g., 'c3').")
    p_dual.add_argument("--xlabel", default=None, help="Optional x-axis label.")
    p_dual.add_argument("--ylabel1", default=None, help="Optional left y-axis label.")
    p_dual.add_argument("--ylabel2", default=None, help="Optional right y-axis label.")
    p_dual.set_defaults(_run=plotter_dual_task)

    # ---- tornado ----
    p_tornado = subparsers.add_parser(
        "tornado",
        help="Tornado plot: label + min/max (and optional median). || "
             "reaxkit plotter tornado --file summary.txt --label c1 --min c2 --max c3 --median c4 --top 10 --save tornado.png || "
             "reaxkit plotter tornado --file summary.txt --label c1 --min c2 --max c3",
    )
    _add_common_io_args(p_tornado)
    p_tornado.add_argument("--label", required=True, help="Column for labels (e.g., 'c1').")
    p_tornado.add_argument("--min", required=True, help="Column for minimum values (e.g., 'c2').")
    p_tornado.add_argument("--max", required=True, help="Column for maximum values (e.g., 'c3').")
    p_tornado.add_argument("--median", default=None, help="Optional column for median values (e.g., 'c4').")
    p_tornado.add_argument("--top", type=int, default=0, help="Show only top-N widest bars (0 = all).")
    p_tornado.add_argument("--vline", type=float, default=None, help="Optional vertical reference line (e.g., 0.0).")
    p_tornado.add_argument("--xlabel", default=None, help="Optional x-axis label.")
    p_tornado.add_argument("--ylabel", default=None, help="Optional y-axis label.")
    p_tornado.set_defaults(_run=plotter_tornado_task)

    # ---- scatter3d ----
    p_scatter3d = subparsers.add_parser(
        "scatter3d",
        help="3D scatter of (x,y,z) points colored by a value. || "
             "reaxkit plotter scatter3d --file summary.txt --x c1 --y c2 --z c3 --value c4 --save 3dscatter.png",
    )
    _add_common_io_args(p_scatter3d)
    p_scatter3d.add_argument("--x", required=True, help="x coordinate column (e.g., 'c1').")
    p_scatter3d.add_argument("--y", required=True, help="y coordinate column (e.g., 'c2').")
    p_scatter3d.add_argument("--z", required=True, help="z coordinate column (e.g., 'c3').")
    p_scatter3d.add_argument("--value", required=True, help="Value column for coloring (e.g., 'c4').")
    p_scatter3d.set_defaults(_run=plotter_scatter3d_task)

    # ---- heatmap2d ----
    p_heatmap2d = subparsers.add_parser(
        "heatmap2d",
        help="2D heatmap from 3D coords + values (projection plane selectable). || "
             "reaxkit plotter heatmap2d --file summary.txt --x c1 --y c2 --z c3 --value c4 --plane xz "
             "--bins 100,80 --save heat_xz.png",
    )
    _add_common_io_args(p_heatmap2d)
    p_heatmap2d.add_argument("--x", required=True, help="x coordinate column (e.g., 'c1').")
    p_heatmap2d.add_argument("--y", required=True, help="y coordinate column (e.g., 'c2').")
    p_heatmap2d.add_argument("--z", required=True, help="z coordinate column (e.g., 'c3').")
    p_heatmap2d.add_argument("--value", required=True, help="value column to aggregate (e.g., 'c4').")
    p_heatmap2d.add_argument("--plane", choices=["xy", "xz", "yz"], default="xy",
        help="Projection plane for heatmap (default: xy).",
    )
    p_heatmap2d.add_argument("--bins", default="50",
        help="Grid resolution: int (e.g., 50) or 'nx,ny' (e.g., '50,100').",
    )
    p_heatmap2d.set_defaults(_run=plotter_heatmap2d_task)
