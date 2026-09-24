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
    values = [
        *[f"{label}_potential (V)" for label in ("internal", "external", "total_local")],
        *[f"{label}_electric_field_{component} ({unit})"
          for label in ("internal", "external", "total_local")
          for unit in ("V/angstrom", "MV/cm") for component in (*"xyz", "magnitude")],
    ]
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
    for label in ("internal", "external", "total_local"):
        vector = output[[f"{label}_electric_field_{axis} (V/angstrom)" for axis in "xyz"]].to_numpy()
        output[f"magnitude_of_average_{label}_electric_field (V/angstrom)"] = np.linalg.norm(vector, axis=1)
        output[f"magnitude_of_average_{label}_electric_field (MV/cm)"] = (
            output[f"magnitude_of_average_{label}_electric_field (V/angstrom)"] * 100.0
        )
    return output.drop(columns="_bin")


def plot_binned_frame(table: pd.DataFrame, axes: str, output: str | Path, *, component: str,
                      units: str, vmin: float | None = None, vmax: float | None = None,
                      dpi: int = 180) -> Path:
    import matplotlib.pyplot as plt
    axes = normalize_axes(axes)
    suffix = "MV/cm" if units == "mv/cm" else "V/angstrom"
    column = ("magnitude_of_average_total_local_electric_field" if component == "magnitude" else
              "total_local_electric_field_magnitude" if component == "mean-magnitude" else
              f"total_local_electric_field_{component}")
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


def _center_edges(values: np.ndarray) -> np.ndarray:
    centers = np.asarray(values, dtype=float)
    if centers.ndim != 1 or not len(centers) or not np.isfinite(centers).all():
        raise ValueError("Kymograph coordinates must be a non-empty finite 1D array.")
    if len(centers) == 1:
        return np.asarray([centers[0] - 0.5, centers[0] + 0.5])
    differences = np.diff(centers)
    if np.any(differences <= 0):
        raise ValueError("Kymograph coordinates must be strictly increasing.")
    middle = centers[:-1] + differences / 2.0
    return np.concatenate(([centers[0] - differences[0] / 2.0], middle,
                           [centers[-1] + differences[-1] / 2.0]))


def plot_binned_kymograph(table: pd.DataFrame, axis_name: str, spatial_edges: np.ndarray,
                          output: str | Path, *, value_column: str,
                          time_column: str = "frame_index", title: str | None = None,
                          cmap: str = "viridis", vmin: float | None = None,
                          vmax: float | None = None, dpi: int = 180) -> Path:
    """Plot one binned scalar versus frame/iteration and 1D spatial position."""
    import matplotlib.pyplot as plt
    axis_name = normalize_axes(axis_name)
    if len(axis_name) != 1:
        raise ValueError("Kymographs require exactly one bin axis.")
    if time_column not in {"frame_index", "iter"}:
        raise ValueError("Kymograph time_column must be frame_index or iter.")
    if value_column not in table.columns:
        raise KeyError(f"Kymograph value column {value_column!r} was not found.")
    times = np.sort(table[time_column].dropna().unique().astype(float))
    n_bins = len(spatial_edges) - 1
    bin_indices = np.arange(n_bins, dtype=int)
    values = (table.pivot_table(index=f"bin_{axis_name}_index", columns=time_column,
                                values=value_column, aggfunc="mean")
              .reindex(index=bin_indices, columns=times).to_numpy(dtype=float))
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9.0, 5.8))
    image = axis.pcolormesh(_center_edges(times), np.asarray(spatial_edges, dtype=float), values,
                            shading="flat", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xlabel("source frame" if time_column == "frame_index" else "iteration")
    axis.set_ylabel(f"{axis_name} (angstrom)")
    axis.set_title(title or f"{value_column} kymograph along {axis_name}")
    figure.colorbar(image, ax=axis).set_label(value_column)
    figure.tight_layout(); figure.savefig(path, dpi=int(dpi), bbox_inches="tight"); plt.close(figure)
    return path


__all__ = ["bin_probe_table", "global_edges", "normalize_axes", "plot_binned_frame",
           "plot_binned_kymograph"]
