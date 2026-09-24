"""Two- and three-dimensional plotting for site-resolved wurtzite polarity."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


PLOT_COLUMNS = {
    "polarity": "polarity",
    "eta": "eta_c (e*angstrom)",
    "delta": "delta_eff (angstrom)",
    "basal": "mean_basal_bond_c (angstrom)",
    "basal-difference": "mean_basal_bond_c_change_from_frame_0 (angstrom)",
}


def plot_site_resolved_polarity(
    table: pd.DataFrame,
    output_directory: str | Path,
    *,
    plane: str | None = None,
    value: str = "polarity",
    color_scale: str = "global",
    slice_range: Sequence[float] | None = None,
    marker_size: float = 12.0,
    dpi: int = 180,
    view_elevation: float = 18.0,
    view_azimuth: float = -60.0,
) -> list[Path]:
    """Write one 3D or projected 2D site plot per trajectory frame."""

    if table.empty:
        raise ValueError("Cannot plot an empty polarity table.")
    selected_plane = None if plane is None else str(plane).lower()
    if selected_plane not in {None, "xy", "xz", "yz"}:
        raise ValueError("plane must be None, 'xy', 'xz', or 'yz'.")
    if value not in PLOT_COLUMNS:
        raise ValueError(f"value must be one of: {', '.join(PLOT_COLUMNS)}.")
    if color_scale not in {"global", "frame"}:
        raise ValueError("color_scale must be 'global' or 'frame'.")
    if not np.isfinite(marker_size) or marker_size <= 0.0:
        raise ValueError("marker_size must be positive and finite.")
    if int(dpi) < 1:
        raise ValueError("dpi must be positive.")
    if not np.isfinite(view_elevation) or not -90.0 <= view_elevation <= 90.0:
        raise ValueError("view_elevation must be between -90 and 90 degrees.")
    if not np.isfinite(view_azimuth):
        raise ValueError("view_azimuth must be finite.")

    is_3d = selected_plane is None
    omitted_axis = None if is_3d else ({"x", "y", "z"} - set(selected_plane)).pop()
    limits = None if slice_range is None else np.asarray(slice_range, dtype=float)
    if is_3d and limits is not None:
        raise ValueError("slice_range requires a 2D plane.")
    if limits is not None and (
        limits.shape != (2,) or not np.isfinite(limits).all() or limits[1] <= limits[0]
    ):
        raise ValueError("slice_range must contain finite MIN MAX values with MIN < MAX.")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, TwoSlopeNorm
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Plot generation requires matplotlib; install reaxkit[plot].") from exc

    value_column = PLOT_COLUMNS[value]
    coordinates = {axis: f"site_{axis} (angstrom)" for axis in "xyz"}
    scale_table = table
    if limits is not None:
        omitted = scale_table[coordinates[omitted_axis]].to_numpy(dtype=float)
        scale_table = scale_table[(omitted >= limits[0]) & (omitted <= limits[1])]
    finite = (
        scale_table["has_four_neighbors"].astype(bool)
        & np.isfinite(scale_table[value_column].to_numpy(dtype=float))
    )
    global_values = scale_table.loc[finite, value_column].to_numpy(dtype=float)

    def continuous_norm(values: np.ndarray):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        lower, upper = ((float(values.min()), float(values.max())) if values.size else (-1.0, 1.0))
        if lower == upper:
            padding = max(abs(lower) * 0.05, 1.0e-12)
            lower, upper = lower - padding, upper + padding
        if value == "basal-difference":
            lower, upper = min(lower, 0.0), max(upper, 0.0)
            if lower == 0.0:
                lower = -max(abs(upper) * 0.05, 1.0e-12)
            if upper == 0.0:
                upper = max(abs(lower) * 0.05, 1.0e-12)
            return TwoSlopeNorm(vmin=lower, vcenter=0.0, vmax=upper)
        return Normalize(vmin=lower, vmax=upper)

    if value == "polarity":
        color_map = ListedColormap(["red", "lightgray", "blue"])
        global_norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], color_map.N)
        ticks = [-1, 0, 1]
    else:
        color_map = plt.get_cmap("RdBu").copy()
        global_norm = continuous_norm(global_values)
        ticks = None

    def expanded(values: np.ndarray) -> tuple[float, float]:
        lower, upper = float(np.nanmin(values)), float(np.nanmax(values))
        padding = 0.02 * (upper - lower) if upper > lower else max(0.5, abs(lower) * 0.02)
        return lower - padding, upper + padding

    coordinate_limits = {
        axis: expanded(table[column].to_numpy(dtype=float)) for axis, column in coordinates.items()
    }
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames = np.sort(table["frame_index"].astype(int).unique())
    width = max(6, len(str(int(frames[-1]))))
    saved: list[Path] = []
    for frame in frames:
        frame_table = table[table["frame_index"].astype(int).eq(frame)].copy()
        if limits is not None:
            omitted = frame_table[coordinates[omitted_axis]].to_numpy(dtype=float)
            frame_table = frame_table[(omitted >= limits[0]) & (omitted <= limits[1])]
        complete = frame_table[
            frame_table["has_four_neighbors"].astype(bool)
            & np.isfinite(frame_table[value_column].to_numpy(dtype=float))
        ]
        incomplete = frame_table[~frame_table.index.isin(complete.index)]
        frame_norm = (
            continuous_norm(complete[value_column].to_numpy(dtype=float))
            if value != "polarity" and color_scale == "frame"
            else global_norm
        )
        if is_3d:
            figure = plt.figure(figsize=(9.0, 7.5), constrained_layout=True)
            axis = figure.add_subplot(111, projection="3d")
            axis.view_init(elev=view_elevation, azim=view_azimuth)
        else:
            figure, axis = plt.subplots(figsize=(9.0, 6.5), constrained_layout=True)
            horizontal, vertical = selected_plane
        if not incomplete.empty:
            if is_3d:
                axis.scatter(*(incomplete[coordinates[a]] for a in "xyz"), c="0.75", s=marker_size)
            else:
                axis.scatter(
                    incomplete[coordinates[horizontal]], incomplete[coordinates[vertical]],
                    c="0.75", s=marker_size,
                )
        scatter_args = dict(c=complete[value_column], cmap=color_map, norm=frame_norm, s=marker_size)
        if is_3d:
            mappable = axis.scatter(*(complete[coordinates[a]] for a in "xyz"), **scatter_args)
            for name in "xyz":
                getattr(axis, f"set_{name}label")(f"{name} (angstrom)")
                getattr(axis, f"set_{name}lim")(*coordinate_limits[name])
            axis.set_box_aspect(tuple(coordinate_limits[a][1] - coordinate_limits[a][0] for a in "xyz"))
            view = "3D"
        else:
            mappable = axis.scatter(
                complete[coordinates[horizontal]], complete[coordinates[vertical]], **scatter_args
            )
            axis.set_xlabel(f"{horizontal} (angstrom)")
            axis.set_ylabel(f"{vertical} (angstrom)")
            axis.set_xlim(*coordinate_limits[horizontal])
            axis.set_ylim(*coordinate_limits[vertical])
            axis.set_aspect("equal", adjustable="box")
            view = f"{selected_plane.upper()} plane"
        iteration = int(frame_table["iter"].iloc[0]) if not frame_table.empty else frame
        axis.set_title(f"Site-resolved wurtzite {value} -- {view}\nframe {frame}, iteration {iteration}")
        colorbar = figure.colorbar(mappable, ax=axis, ticks=ticks, shrink=0.88)
        colorbar.set_label(value_column)
        if value == "polarity":
            colorbar.set_ticklabels(["DOWN", "unassigned", "UP"])
        destination = output / f"local_polarity_frame_{frame:0{width}d}.png"
        figure.savefig(destination, dpi=int(dpi), bbox_inches="tight")
        plt.close(figure)
        saved.append(destination)
    return saved


__all__ = ["PLOT_COLUMNS", "plot_site_resolved_polarity"]
