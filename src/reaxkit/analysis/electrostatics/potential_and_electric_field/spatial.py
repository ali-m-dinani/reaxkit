"""Spatial binning and per-frame plotting of local electric fields."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

AXIS_COLUMN = {"x": "x (angstrom)", "y": "y (angstrom)", "z": "z (angstrom)"}


def normalize_axes(value: str) -> str:
    axes = str(value).lower().replace(",", "").strip()
    if not 1 <= len(axes) <= 3 or len(set(axes)) != len(axes) or any(v not in "xyz" for v in axes):
        raise ValueError("Bin axes must contain one to three unique x, y, and z axes.")
    return axes


def global_edges(tables: Sequence[pd.DataFrame], axes: str, bins: Sequence[int]) -> tuple[np.ndarray, ...]:
    axes = normalize_axes(axes)
    counts = tuple(int(value) for value in bins)
    if len(counts) == 1:
        counts *= len(axes)
    if len(counts) != len(axes) or any(value < 1 for value in counts):
        raise ValueError("Provide one positive bin count or one per selected axis.")
    combined = pd.concat(tables, ignore_index=True)
    edges = []
    for axis, count in zip(axes, counts):
        values = combined[AXIS_COLUMN[axis]].to_numpy(dtype=float)
        lower, upper = float(np.nanmin(values)), float(np.nanmax(values))
        if np.isclose(lower, upper):
            lower -= 0.5; upper += 0.5
        edges.append(np.linspace(lower, upper, count + 1))
    return tuple(edges)


def bin_probe_table(table: pd.DataFrame, axes: str, edges: Sequence[np.ndarray]) -> pd.DataFrame:
    axes = normalize_axes(axes)
    samples = table[[AXIS_COLUMN[axis] for axis in axes]].to_numpy(dtype=float)
    shape = tuple(len(edge) - 1 for edge in edges)
    indices = [np.clip(np.digitize(samples[:, dim], edge) - 1, 0, len(edge) - 2)
               for dim, edge in enumerate(edges)]
    linear = np.ravel_multi_index(indices, shape)
    work = table.copy(); work["_bin"] = linear
    values = ["potential (V)", "electric_field_x (V/angstrom)", "electric_field_y (V/angstrom)",
              "electric_field_z (V/angstrom)", "electric_field_magnitude (V/angstrom)",
              "electric_field_x (MV/cm)", "electric_field_y (MV/cm)",
              "electric_field_z (MV/cm)", "electric_field_magnitude (MV/cm)"]
    grouped = work.groupby("_bin", sort=True)
    output = grouped[values].mean().reset_index()
    output.insert(1, "atom_count", grouped.size().to_numpy())
    multi = np.asarray(np.unravel_index(output["_bin"].to_numpy(dtype=int), shape)).T
    for dim, axis in enumerate(axes):
        idx = multi[:, dim]; edge = np.asarray(edges[dim])
        output[f"bin_{axis}_index"] = idx
        output[f"{axis}_lower (angstrom)"] = edge[idx]
        output[f"{axis}_center (angstrom)"] = (edge[idx] + edge[idx + 1]) / 2
        output[f"{axis}_upper (angstrom)"] = edge[idx + 1]
    vector = output[[f"electric_field_{axis} (V/angstrom)" for axis in "xyz"]].to_numpy()
    output["magnitude_of_average_field (V/angstrom)"] = np.linalg.norm(vector, axis=1)
    output["magnitude_of_average_field (MV/cm)"] = output["magnitude_of_average_field (V/angstrom)"] * 100.0
    return output.drop(columns="_bin")


def plot_binned_frame(table: pd.DataFrame, axes: str, output: str | Path, *, component: str,
                      units: str, vmin: float | None = None, vmax: float | None = None,
                      dpi: int = 180) -> Path:
    import matplotlib.pyplot as plt
    axes = normalize_axes(axes)
    suffix = "MV/cm" if units == "mv/cm" else "V/angstrom"
    column = ("magnitude_of_average_field" if component == "magnitude" else
              "electric_field_magnitude" if component == "mean-magnitude" else f"electric_field_{component}")
    value_column = f"{column} ({suffix})"
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(8, 6))
    if len(axes) == 1:
        axis = figure.add_subplot(111)
        axis.plot(table[f"{axes[0]}_center (angstrom)"], table[value_column], marker="o", markersize=3)
        axis.set_xlabel(f"{axes[0]} (angstrom)"); axis.set_ylabel(value_column)
        if vmin is not None or vmax is not None: axis.set_ylim(vmin, vmax)
    elif len(axes) == 2:
        axis = figure.add_subplot(111)
        x, y = axes
        grid = table.pivot(index=f"{y}_center (angstrom)", columns=f"{x}_center (angstrom)", values=value_column)
        image = axis.imshow(grid.to_numpy(), origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
                            extent=[grid.columns.min(), grid.columns.max(), grid.index.min(), grid.index.max()])
        axis.set_xlabel(f"{x} (angstrom)"); axis.set_ylabel(f"{y} (angstrom)")
        figure.colorbar(image, ax=axis, label=value_column)
    else:
        axis = figure.add_subplot(111, projection="3d")
        coords = [table[f"{value}_center (angstrom)"] for value in axes]
        points = axis.scatter(*coords, c=table[value_column], cmap="viridis", vmin=vmin, vmax=vmax, s=18)
        axis.set_xlabel(f"{axes[0]} (angstrom)"); axis.set_ylabel(f"{axes[1]} (angstrom)"); axis.set_zlabel(f"{axes[2]} (angstrom)")
        figure.colorbar(points, ax=axis, label=value_column, shrink=0.75)
    figure.tight_layout(); figure.savefig(path, dpi=int(dpi)); plt.close(figure)
    return path


__all__ = ["bin_probe_table", "global_edges", "normalize_axes", "plot_binned_frame"]
