"""all types of plots used in reaxkit"""
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
    """determines if a plot is going to be only shown or saved as png.
    Save to a file or show:
       - If save_dir looks like a file path with an image extension, save exactly there.
       - Else, treat save_dir as a directory and save as <dir>/<filename>.png
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
    """Create a single matplotlib figure for quick plotting of one or multiple data series.

    Supports both simple (x, y) inputs and flexible multi-series plotting using a
    list of dictionaries. Can render line or scatter plots, add labeled horizontal
    reference lines, and optionally save or display the figure automatically.

    Typical use cases include visualizing time series, bond-order traces, or
    diagnostic overlays in ReaxFF analyses.

    Behavior
    --------
    - If `series` is given, each entry should be a mapping with keys like
      "x", "y", "label", "marker", "linewidth", "markersize", "alpha".
    - If `series` is None, falls back to simple (x, y) plotting.
    - Horizontal lines (`hlines`) can be:
        * a float → y-value for dashed line,
        * a tuple → (y, label),
        * a dict → {"y": value, "label": ..., "linestyle": ..., "linewidth": ..., "alpha": ...}.
    - The `plot_type` argument controls whether to use `plot` (line) or `scatter`.

    Notes
    -----
    - Automatically calls `_save_or_show(fig, save, title)`:
        * if `save=None` → shows interactively,
        * if `save` is a directory → saves figure in that directory,
        * if `save` is a full path → saves to that file.
    - Automatically applies `tight_layout()` for clean spacing.
    - Returns the matplotlib Figure for further customization if desired.
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
    """Plot a continuous line with directional arrows showing progression along (x, y).

    The function connects consecutive (x, y) points with a line and overlays
    small arrows to indicate the forward direction of traversal, making it useful
    for visualizing trajectories, energy paths, or time evolution of quantities.

    Behavior
    --------
    - Draws the main path using `ax.plot` with configurable color and linestyle.
    - Uses `ax.quiver` to overlay directional arrows along the path, computed
      from the point-to-point differences (dx, dy).
    - Supports optional grid, horizontal reference line (`hline`), and axis limits.
    - Can display interactively or save to file via `_save_or_show`.

    Notes
    -----
    - Arrows are drawn between successive data points, with uniform scaling for clarity.
    - The default arrow width is small (0.003) for dense trajectories; increase for clarity.
    - The function does not return a value — it displays or saves the plot directly.
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
    """Plot two datasets sharing a common x-axis but with separate left and right y-axes.

    This function is useful when comparing two quantities with different units or
    magnitudes, such as energy vs. temperature or pressure vs. time, on a single plot.

    Behavior
    --------
    - Plots `y1` on the left y-axis (`ax1`) and `y2` on the right y-axis (`ax2`).
    - Each dataset can have its own color, line style, marker, and axis limits.
    - Supports optional horizontal lines on both y-axes (`hline1`, `hline2`)
      and a vertical reference line (`vline`).
    - Shared grid and title are applied to the figure.
    - Uses `_save_or_show(fig, save, title)` to display or save automatically.

    Notes
    -----
    - Ideal for visualizing related but differently scaled quantities.
    - The left and right axes are colored to match their corresponding datasets.
    - If both y-axes have horizontal reference lines, each uses its respective color.
    - Returns no value; the figure is shown or saved directly.
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
    """Create a horizontal **tornado plot** showing parameter sensitivity ranges.

    Each label represents a variable or factor with a corresponding minimum and
    maximum value, plotted as a filled horizontal bar. The bar length indicates
    the impact range, and an optional median marker (`*`) highlights a central estimate.

    Behavior
    --------
    - Bars are sorted by total span (`max - min`) in descending order.
    - If `top` > 0, only the top N widest bars are shown.
    - Bars are colored according to their position relative to `vline`:
        * Entirely left of `vline`  → `left_color`
        * Entirely right of `vline` → `right_color`
        * Crossing `vline`          → split bar (left/right segments)
    - If `vline` is None, all bars are drawn in a neutral gray.
    - Optional median markers (`median_vals`) are shown as black asterisks (`*`).
    - Empty data gracefully shows a “No data to plot” placeholder.

    Notes
    -----
    - Automatically adjusts figure height based on the number of bars.
    - The y-axis is inverted so the widest bar appears at the top.
    - Useful for **sensitivity analysis**, **uncertainty visualization**, or
      **parameter impact ranking** in simulations.
    - Uses `_save_or_show(fig, save, title)` to display or save automatically.
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
    figsize: Tuple[float, float] = (6.5, 6.0),
    elev: float = 22.0,
    azim: float = 38.0,
    save: Optional[Union[str, Path]] = None,   # dir or full path w/ extension
    show_colorbar: bool = True,
    show_message: bool = True,
):
    """Render a 3D scatter plot of atomic coordinates, colored by a per-atom property such as partial charges.

    Each point corresponds to an atom located at (x, y, z), and its color encodes
    a scalar quantity such as partial charge, potential energy, or bond order sum.

    Behavior
    --------
    - Displays a 3D scatter plot (`matplotlib.axes._subplots.Axes3DSubplot`) with
      colors mapped through `values` using the specified colormap (`cmap`).
    - Optional color range control via `vmin` and `vmax`.
    - Camera view set by elevation (`elev`) and azimuth (`azim`).
    - Marker transparency and size controlled via `alpha` and `s`.
    - Can show or hide the colorbar (`show_colorbar`).
    - Uses `_save_or_show(fig, save, title)` to either display interactively
      or save to a directory/file path.

    Notes
    -----
    - `coords` must be an array of shape (N, 3).
    - `values` must be of length N; each entry corresponds to one atom.
    - Automatically scales color limits if `vmin` and `vmax` are not given.
    - Useful for visualizing **charge distributions**, **defect localizations**,
      or **per-atom properties** from ReaxFF simulations.
    - If `show_message=True`, prints a short summary (e.g., number of points plotted).
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
    """Project 3D points onto a 2D plane and aggregate 'values' over a regular grid used for plotting a property across 2D planes.

    Returns
    -------
    fig : matplotlib.figure.Figure
    grid : (ny, nx) np.ndarray with aggregated values (NaN where empty)
    xedges, yedges : np.ndarray bin edges (len = nx+1, ny+1)
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
