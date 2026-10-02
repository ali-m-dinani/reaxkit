"""Build a radius-by-frame matrix without rescaling its values."""

from __future__ import annotations

import numpy as np
import pandas as pd


def kymograph_grid(rows, x_col="frame_index", y_col="r", color_col="g"):
    """Return sorted coordinates and values, requiring a common radial grid."""
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No data available for a kymograph.")
    columns = [x_col, y_col, color_col]
    numeric = table[columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Kymograph coordinates and values must be finite.")
    radial_grid = None
    coordinates = []
    values = []
    for coordinate, group in numeric.groupby(x_col, sort=True):
        group = group.sort_values(y_col)
        radii = group[y_col].to_numpy()
        if group[y_col].duplicated().any():
            raise ValueError("Kymograph requires unique radial bins in each frame.")
        if radial_grid is None:
            radial_grid = radii
        elif radii.shape != radial_grid.shape or not np.allclose(radii, radial_grid, rtol=1e-7, atol=1e-10):
            raise ValueError("R grids differ between frames; use fixed bins and r_max valid for every frame.")
        coordinates.append(coordinate)
        values.append(group[color_col].to_numpy())
    return np.asarray(coordinates), radial_grid, np.asarray(values).T
