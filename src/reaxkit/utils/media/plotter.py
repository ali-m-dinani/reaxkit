"""
Plotting utilities for ReaxKit.

This module provides a collection of lightweight, reusable plotting helpers
built on Matplotlib for visualizing ReaxFF simulation data. The functions
support common use cases such as time-series plots, multi-axis comparisons,
stacked subplots, 3D atom visualizations, sensitivity (tornado) plots, and
2D projections of 3D data.

All helpers share a consistent save/show behavior and are designed to be
used directly by analyzers and workflows.
"""

from __future__ import annotations
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, Callable, Mapping, Any, Dict

# ---------- Helpers ----------
def _save_or_show(
    fig: plt.Figure,
    save_dir: Optional[Union[str, Path]],
    filename: str,
    show_message: bool = True,
) -> None:
    """
    Save a figure to disk or display it interactively.

    If ``save_dir`` is provided, the figure is saved either to the given
    file path or into the specified directory using ``filename``. If
    ``save_dir`` is ``None``, the figure is shown interactively.
    """
    if save_dir:
        save_path = Path(save_dir)
        # Known image/document extensions
        exts = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".tif", ".tiff", ".bmp"}
        if save_path.suffix.lower() in exts:
            # save_dir is a full file path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            out_file = save_path
        else:
            # save_dir is a directory; build filename.png inside it
            save_path.mkdir(parents=True, exist_ok=True)
            safe_name = filename.replace(" ", "_")
            out_file = save_path / f"{safe_name}.png"

        fig.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[Done] saved plot to {out_file}") if show_message else None
    else:
        plt.show()


# ---------- Plots ----------
def single_plot(
    x: Optional[Sequence[float]] = None,
    y: Optional[Sequence[float]] = None,
    *,
    series: Optional[Sequence[Mapping[str, Any]]] = None,
    hlines: Optional[Sequence[Union[float, tuple, Mapping[str, Any]]]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    save: Optional[Union[str, Path]] = None,
    legend: bool = False,
    figsize: tuple[float, float] = (8.0, 3.2),
    plot_type: str = "line",
) -> plt.Figure:
    """
    Create a single plot for one or multiple data series.

    This function supports both simple ``(x, y)`` inputs and flexible
    multi-series plotting using a list of dictionaries. It can render
    line or scatter plots, add horizontal reference lines, and
    automatically save or display the figure.

    Parameters
    ----------
    x, y : sequence of float, optional
        Data to plot when using the simple API.
    series : sequence of mapping, optional
        Multi-series specification with keys such as ``x``, ``y``,
        ``label``, ``marker``, and ``linewidth``.
    hlines : sequence, optional
        Horizontal reference lines specified as floats, tuples, or dicts.
    title, xlabel, ylabel : str, optional
        Plot title and axis labels.
    save : str or Path, optional
        Output directory or file path. If not provided, the plot is shown.
    legend : bool, optional
        Whether to display a legend.
    figsize : tuple of float, optional
        Figure size.
    plot_type : {'line', 'scatter'}, optional
        Type of plot to generate.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """


    fig, ax = plt.subplots(figsize=figsize)

    def _plot(ax, x, y, label=None, **kwargs):
        if plot_type == "scatter":
            ax.scatter(x, y, label=label, **kwargs)
        else:
            ax.plot(x, y, label=label, **kwargs)

    # Multiple series
    if series is not None:
        for s in series:
            sx = s.get("x")
            sy = s.get("y")
            if sx is None or sy is None:
                continue
            lbl = s.get("label")
            lw = s.get("linewidth", 1.2)
            mk = s.get("marker", "." if plot_type == "scatter" else None)
            ms = s.get("markersize", 4)
            al = s.get("alpha", 1.0)

            # Map marker size kw correctly per plot type
            kwargs = dict(linewidth=lw, marker=mk, alpha=al)
            if plot_type == "scatter":
                kwargs["s"] = ms          # scatter uses 's'
            else:
                kwargs["markersize"] = ms # line plot uses 'markersize'

            _plot(ax, sx, sy, label=lbl, **kwargs)
    else:
        if x is None or y is None:
            raise ValueError("Provide (x, y) or 'series=[...]'.")
        if plot_type == "scatter":
            ax.scatter(x, y, label=None)
        else:
            ax.plot(x, y, label=None)

    # Horizontal lines (unchanged) ...
    if hlines:
        for h in hlines:
            if isinstance(h, Mapping):
                yv = h.get("y")
                if yv is None:
                    continue
                lbl = h.get("label")
                ls = h.get("linestyle", "--")
                lw = h.get("linewidth", 1.0)
                al = h.get("alpha", 1.0)
                ax.axhline(yv, linestyle=ls, linewidth=lw, alpha=al, color="gray", label=lbl)
            elif isinstance(h, tuple):
                yv, lbl = h[0], (h[1] if len(h) > 1 else None)
                ax.axhline(yv, linestyle="--", linewidth=1.0, alpha=1.0, color="gray", label=lbl)
            else:
                ax.axhline(float(h), linestyle="--", linewidth=1.0, alpha=1.0, color="gray")

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend()

    fig.tight_layout()
    _save_or_show(fig, save, title or "single_plot")
    return fig


def directed_plot(
    x: Sequence[float],
    y: Sequence[float],
    *,
    figsize: Tuple[float, float] = (10, 6),
    title: str = '',
    xlabel: str = '',
    ylabel: str = '',
    color: str = 'blue',
    linestyle: str = '-',
    arrow_color: str = 'red',
    arrow_width: float = 0.003,
    grid: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    hline: Optional[float] = None,
    hline_kwargs: Optional[Dict] = None,
    legend: bool = False,
    save: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot a 2D path with directional arrows.

    This helper visualizes progression along a trajectory by drawing a
    continuous line through ``(x, y)`` points and overlaying arrows that
    indicate direction. It is useful for trajectories, energy paths, or
    ordered parameter sweeps.

    Parameters
    ----------
    x, y : sequence of float
        Path coordinates.
    title, xlabel, ylabel : str, optional
        Plot title and axis labels.
    save : str or Path, optional
        Output directory or file path.

    Returns
    -------
    None
    """
    dx = np.diff(x)
    dy = np.diff(y)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, linestyle=linestyle, color=color, label='Path')
    ax.quiver(
        x[:-1], y[:-1], dx, dy,
        angles='xy', scale_units='xy', scale=1,
        color=arrow_color, width=arrow_width
    )

    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if grid:
        ax.grid(True)
    if hline is not None:
        params = {'color': 'black', 'linestyle': '--', 'linewidth': 1}
        if hline_kwargs:
            params.update(hline_kwargs)
        ax.axhline(hline, **params)
    if legend:
        ax.legend()

    _save_or_show(fig, save, title or 'directed_plot')


def dual_yaxis_plot(
    x: Sequence[float],
    y1: Sequence[float],
    y2: Sequence[float],
    *,
    figsize: Tuple[float, float] = (10, 6),
    title: str = '',
    xlabel: str = '',
    ylabel1: str = '',
    ylabel2: str = '',
    color1: str = 'blue',
    linestyle1: str = '-',
    marker1: str = '',
    color2: str = 'green',
    linestyle2: str = '--',
    marker2: str = '',
    xlim: Optional[Tuple[float, float]] = None,
    ylim1: Optional[Tuple[float, float]] = None,
    ylim2: Optional[Tuple[float, float]] = None,
    grid: bool = False,
    hline1: Optional[float] = None,
    hline1_kwargs: Optional[Dict] = None,
    hline2: Optional[float] = None,
    hline2_kwargs: Optional[Dict] = None,
    vline: Optional[float] = None,
    vline_kwargs: Optional[Dict] = None,
    save: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot two datasets against a shared x-axis with separate y-axes.

    This function is intended for comparing quantities with different
    units or magnitudes on the same plot (e.g., energy vs temperature).

    Parameters
    ----------
    x : sequence of float
        Shared x-axis values.
    y1, y2 : sequence of float
        Data for the left and right y-axes.
    title, xlabel, ylabel1, ylabel2 : str, optional
        Labels and title.
    save : str or Path, optional
        Output directory or file path.

    Returns
    -------
    None
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(x, y1, linestyle=linestyle1, marker=marker1, color=color1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color=color1)
    if xlim:
        ax1.set_xlim(xlim)
    if ylim1:
        ax1.set_ylim(ylim1)
    if grid:
        ax1.grid(True)
    if hline1 is not None:
        params = {'color': color1, 'linestyle': '--', 'linewidth': 1}
        if hline1_kwargs:
            params.update(hline1_kwargs)
        ax1.axhline(hline1, **params)

    ax2 = ax1.twinx()
    ax2.plot(x, y2, linestyle=linestyle2, marker=marker2, color=color2)
    ax2.set_ylabel(ylabel2, color=color2)
    if ylim2:
        ax2.set_ylim(ylim2)
    if hline2 is not None:
        params = {'color': color2, 'linestyle': '--', 'linewidth': 1}
        if hline2_kwargs:
            params.update(hline2_kwargs)
        ax2.axhline(hline2, **params)

    if vline is not None:
        params = {'color': 'black', 'linestyle': ':', 'linewidth': 1}
        if vline_kwargs:
            params.update(vline_kwargs)
        ax1.axvline(vline, **params)

    fig.suptitle(title)
    _save_or_show(fig, save, title or 'dual_yaxis_plot')


def multi_subplots(
    subplots: Sequence[Sequence[Mapping[str, Any]]],
    *,
    title: Optional[Union[str, Sequence[Optional[str]]]] = None,
    xlabel: Optional[Union[str, Sequence[Optional[str]]]] = None,
    ylabel: Optional[Union[str, Sequence[Optional[str]]]] = None,
    sharex: bool = False,
    sharey: bool = False,
    legend: bool = True,
    grid: bool = False,
    figsize: tuple[float, float] = (8.0, 6.0),
    save: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Create multiple vertically stacked subplots.

    Each subplot accepts the same series specification used by
    ``single_plot``, allowing consistent plotting across panels.
    Titles and axis labels may be shared or specified per subplot.

    Parameters
    ----------
    subplots : sequence of sequence of dict
        Series definitions for each subplot.
    title, xlabel, ylabel : str or sequence of str, optional
        Global or per-subplot titles and labels.
    save : str or Path, optional
        Output directory or file path.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    import matplotlib.pyplot as plt

    nplots = len(subplots)
    if nplots == 0:
        print("multi_subplots: no subplot data provided.")
        return None  # type: ignore[return-value]

    def _normalize_seq(
        val: Optional[Union[str, Sequence[Optional[str]]]],
        n: int,
    ) -> list[Optional[str]]:
        """Turn val into a list of length n.

        - None → [None] * n
        - str → [str] * n
        - sequence:
            * len == 1 → repeat for all
            * len == n → use as-is
            * otherwise → error
        """
        if val is None:
            return [None] * n
        if isinstance(val, (list, tuple)):
            if len(val) == 1:
                return [val[0]] * n
            if len(val) != n:
                raise ValueError(
                    f"Expected sequence of length 1 or {n}, got length {len(val)}"
                )
            return list(val)
        # single string
        return [val] * n

    # Handle global vs per-subplot title
    if isinstance(title, (list, tuple)):
        per_titles = _normalize_seq(title, nplots)
        global_title = None
    else:
        per_titles = [None] * nplots
        global_title = title

    xlabels = _normalize_seq(xlabel, nplots)
    ylabels = _normalize_seq(ylabel, nplots)

    fig, axes = plt.subplots(
        nplots,
        1,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx >= nplots:
            break

        # Plot series for this subplot
        for series in subplots[idx]:
            x = series.get("x")
            y = series.get("y")
            if x is None or y is None:
                continue
            label = series.get("label")
            ax.plot(x, y, label=label)

        # Per-subplot labels/titles
        if ylabels[idx]:
            ax.set_ylabel(ylabels[idx])
        if xlabels[idx]:
            ax.set_xlabel(xlabels[idx])
        if per_titles[idx]:
            ax.set_title(per_titles[idx])

        if grid:
            ax.grid(True, alpha=0.3)
        if legend:
            ax.legend(fontsize=9)

    # Global suptitle if provided as a single string
    if global_title:
        fig.suptitle(global_title, fontsize=14, y=0.98)

    fig.tight_layout()
    _save_or_show(fig, save, global_title or "multi_subplots")
    return fig




def tornado_plot(
    labels: Sequence[str],
    min_vals: Sequence[float],
    max_vals: Sequence[float],
    *,
    median_vals: Optional[Sequence[float]] = None,
    title: str = "Tornado Plot",
    xlabel: str = "Value",
    ylabel: str = "Value",
    save: Optional[Union[str, Path]] = None,
    top: int = 0,
    vline: Optional[float] = None,
    left_color="#1F77B4",
    right_color=(225 / 255, 113 / 255, 29 / 255)  # RGB tuple
) -> None:
    """
    Create a tornado plot to visualize sensitivity or uncertainty ranges.

    Each label is represented by a horizontal bar spanning from a minimum
    to a maximum value. Bars are ordered by total span, highlighting the
    most influential parameters.

    Parameters
    ----------
    labels : sequence of str
        Parameter or variable names.
    min_vals, max_vals : sequence of float
        Lower and upper bounds for each parameter.
    median_vals : sequence of float, optional
        Optional central estimates to mark on each bar.
    save : str or Path, optional
        Output directory or file path.

    Returns
    -------
    None
    """
    df = pd.DataFrame({"label": labels, "min": min_vals, "max": max_vals})
    if median_vals is not None:
        if len(median_vals) != len(labels):
            raise ValueError("median_vals must be the same length as labels/min_vals/max_vals")
        df["median"] = list(median_vals)

    df["span"] = df["max"] - df["min"]
    df = df.sort_values("span", ascending=False).reset_index(drop=True)
    if top and top > 0:
        df = df.head(top)

    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
        ax.axis("off")
        _save_or_show(fig, save, filename=title or "tornado_plot")
        return

    fig_height = max(3.2, 0.38 * len(df))
    fig, ax = plt.subplots(figsize=(8.6, fig_height))

    y_positions = np.arange(len(df))
    bar_height = 0.5
    edge_common = dict(edgecolor="black", linewidth=0.5)

    for y, row in zip(y_positions, df.itertuples(index=False)):
        if row.max < row.min:  # guard against swapped inputs
            left, right = row.max, row.min
        else:
            left, right = row.min, row.max

        if vline is None:
            # no reference: draw a single neutral bar
            ax.barh(y=y, width=right - left, left=left, height=bar_height,
                    color="tab:gray", alpha=0.7, **edge_common)
        else:
            if right <= vline:
                # fully left of vline
                ax.barh(y=y, width=right - left, left=left, height=bar_height,
                        color=left_color, alpha=0.8, **edge_common)
            elif left >= vline:
                # fully right of vline
                ax.barh(y=y, width=right - left, left=left, height=bar_height,
                        color=right_color, alpha=0.8, **edge_common)
            else:
                # straddles vline -> split into two bars
                ax.barh(y=y, width=vline - left, left=left, height=bar_height,
                        color=left_color, alpha=0.8, **edge_common)
                ax.barh(y=y, width=right - vline, left=vline, height=bar_height,
                        color=right_color, alpha=0.8, **edge_common)

        # median marker as asterisk
        if "median" in df.columns and not pd.isna(row.median):
            ax.plot(row.median, y, marker="*", markersize=5, color='black', zorder=5)

    if vline is not None:
        ax.axvline(vline, linestyle="--", linewidth=1, color=(66/255, 196/255, 127/255))

        # simple legend patches to match the example look
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor=left_color, edgecolor="black"),
            Patch(facecolor=right_color, edgecolor="black"),
            plt.Line2D([0], [0], marker="*", linestyle="none", color="black",
                       markersize=10),
        ]


    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(df["label"])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()

    _save_or_show(fig, save, filename=title or "tornado_plot")




def scatter3d_points(
    coords: np.ndarray,            # (N, 3) array of XYZ
    values: np.ndarray,            # (N,) array, e.g., partial charges
    *,
    title: str = "atoms (3D)",
    s: float = 8.0,
    alpha: float = 0.9,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "coolwarm",
    figsize: Tuple[float, float] = (7.5, 6.0),
    elev: float = 22.0,
    azim: float = 38.0,
    save: Optional[Union[str, Path]] = None,   # dir or full path w/ extension
    show_colorbar: bool = True,
    show_message: bool = True,
):
    """
    Render a 3D scatter plot of atomic coordinates.

    Points are colored by a scalar per-atom property such as partial
    charge or bond-order sum, enabling spatial visualization of
    atom-resolved quantities.

    Parameters
    ----------
    coords : array-like, shape (N, 3)
        Atomic coordinates.
    values : array-like, shape (N,)
        Scalar values mapped to colors.
    save : str or Path, optional
        Output directory or file path.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    coords = np.asarray(coords, float)
    values = np.asarray(values, float)
    assert coords.ndim == 2 and coords.shape[1] == 3, "coords must be (N,3)"
    assert values.ndim == 1 and len(values) == len(coords), "values must be (N,)"

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=values, cmap=cmap, vmin=vmin, vmax=vmax,
        s=s, alpha=alpha, depthshade=True
    )
    if show_colorbar:
        cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
        cb.set_label("value")

    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    _save_or_show(fig, save, filename=title, show_message =show_message)

    return fig


def heatmap2d_from_3d(
    coords: np.ndarray,                 # (N,3) array of XYZ
    values: np.ndarray,                 # (N,) scalar to aggregate (e.g., partial_charge)
    *,
    plane: str = "xy",                  # "xy", "xz", or "yz"
    bins: Union[int, Tuple[int, int]] = 50,  # grid resolution; int or (nx, ny)
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    agg: Union[str, Callable[[np.ndarray], float]] = "mean",  # "mean","max","min","sum","count" or a callable
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    title: str = "2D aggregated heatmap",
    figsize: Tuple[float, float] = (6.5, 5.5),
    save: Optional[Union[str, Path]] = None,  # dir or full path; if dir, saves PNG
    show_colorbar: bool = True,
    show_message: bool = True,
):
    """
    Project 3D point data onto a 2D plane and aggregate values on a grid.

    This function bins projected coordinates onto a regular grid and
    aggregates per-point values using a specified operation, producing
    a 2D heatmap suitable for planar analysis.

    Parameters
    ----------
    coords : array-like, shape (N, 3)
        3D coordinates.
    values : array-like, shape (N,)
        Values to aggregate.
    plane : {'xy', 'xz', 'yz'}, optional
        Projection plane.
    agg : {'mean', 'max', 'min', 'sum', 'count'} or callable, optional
        Aggregation method.
    save : str or Path, optional
        Output directory or file path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    grid : numpy.ndarray
        Aggregated 2D grid.
    xedges, yedges : numpy.ndarray
        Bin edges along each axis.
    """
    coords = np.asarray(coords, float)
    values = np.asarray(values, float)
    assert coords.ndim == 2 and coords.shape[1] == 3, "coords must be (N,3)"
    assert values.ndim == 1 and len(values) == len(coords), "values must be (N,)"

    # Choose plane projection
    if plane == "xy":
        u, v = coords[:, 0], coords[:, 1]
        xlabel, ylabel = "x (Å)", "y (Å)"
    elif plane == "xz":
        u, v = coords[:, 0], coords[:, 2]
        xlabel, ylabel = "x (Å)", "z (Å)"
    elif plane == "yz":
        u, v = coords[:, 1], coords[:, 2]
        xlabel, ylabel = "y (Å)", "z (Å)"
    else:
        raise ValueError("plane must be one of {'xy','xz','yz'}")

    # Grid resolution
    if isinstance(bins, int):
        nx = ny = int(bins)
    else:
        nx, ny = int(bins[0]), int(bins[1])

    # Ranges
    umin = np.min(u) if xlim is None else xlim[0]
    umax = np.max(u) if xlim is None else xlim[1]
    vmin_edge = np.min(v) if ylim is None else ylim[0]
    vmax_edge = np.max(v) if ylim is None else ylim[1]

    # Bin edges
    xedges = np.linspace(umin, umax, nx + 1)
    yedges = np.linspace(vmin_edge, vmax_edge, ny + 1)

    # Digitize points to cell indices
    # Points exactly on the right/top edge go to the previous bin
    ui = np.clip(np.digitize(u, xedges) - 1, 0, nx - 1)
    vi = np.clip(np.digitize(v, yedges) - 1, 0, ny - 1)

    # Flattened cell id for bincount tricks
    flat_idx = vi * nx + ui
    n_cells = nx * ny

    grid = np.full((ny, nx), np.nan, float)

    # Built-in fast paths
    if isinstance(agg, str):
        agg_lower = agg.lower()
        if agg_lower == "count":
            cnt = np.bincount(flat_idx, minlength=n_cells).astype(float)
            grid = cnt.reshape(ny, nx)
        elif agg_lower in {"sum", "mean"}:
            sumv = np.bincount(flat_idx, weights=values, minlength=n_cells).astype(float)
            cnt = np.bincount(flat_idx, minlength=n_cells).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                if agg_lower == "sum":
                    grid = sumv.reshape(ny, nx)
                else:  # mean
                    grid = (sumv / cnt).reshape(ny, nx)
                    grid[cnt.reshape(ny, nx) == 0] = np.nan
        elif agg_lower in {"max", "min"}:
            # One-pass update
            fill_val = -np.inf if agg_lower == "max" else np.inf
            flat_grid = np.full(n_cells, fill_val, float)
            for idx, val in zip(flat_idx, values):
                if agg_lower == "max":
                    if val > flat_grid[idx]:
                        flat_grid[idx] = val
                else:
                    if val < flat_grid[idx]:
                        flat_grid[idx] = val
            # Convert untouched cells to NaN
            if agg_lower == "max":
                flat_grid[flat_grid == -np.inf] = np.nan
            else:
                flat_grid[flat_grid == np.inf] = np.nan
            grid = flat_grid.reshape(ny, nx)
        else:
            raise ValueError("agg must be one of {'mean','max','min','sum','count'} or a callable")
    else:
        # Callable aggregator: collect values per cell (okay for modest grids)
        buckets: Sequence[list] = [list() for _ in range(n_cells)]
        for idx, val in zip(flat_idx, values):
            buckets[idx].append(val)
        flat_grid = np.full(n_cells, np.nan, float)
        for i, bucket in enumerate(buckets):
            if bucket:
                try:
                    flat_grid[i] = agg(np.asarray(bucket, float))
                except Exception:
                    # Fallback: ignore cell if aggregator fails
                    flat_grid[i] = np.nan
        grid = flat_grid.reshape(ny, nx)

    # Color scale defaults from data if not provided
    cmin = np.nanmin(grid) if vmin is None else vmin
    cmax = np.nanmax(grid) if vmax is None else vmax

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    # extent maps grid cells to physical coordinates
    extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
    im = ax.imshow(
        grid,
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=cmin,
        vmax=cmax,
        cmap=cmap,
        interpolation="nearest",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if show_colorbar:
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(f"{agg if isinstance(agg,str) else 'agg'} of values")

    _save_or_show(fig, save, title.replace(" ", "_"), show_message =show_message)

    return fig, grid, xedges, yedges
