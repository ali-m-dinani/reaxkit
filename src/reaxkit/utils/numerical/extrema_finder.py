"""
1D extrema detection utilities.

This module provides helper functions for identifying extrema in one-dimensional
data series commonly produced by ReaxFF simulations, such as energy profiles,
bond-order trajectories, dipole signals, polarization loops, or field-response
curves.

Typical use cases include:

- locating global or local energy minima and maxima
- identifying peak responses in field-driven simulations
- detecting switching or transition points in time-series data
"""


import numpy as np
import pandas as pd


def get_extrema_points(y_series, x_series, mode='max', chunk_size=None):
    """
    Identify extrema points in a 1D data series.

    This function extracts global or local extrema by locating maximum and/or
    minimum values of a y-series with respect to a corresponding x-series.
    Optionally, the x-axis may be partitioned into windows to detect local
    extrema within each segment.

    Parameters
    ----------
    y_series : pandas.Series or array-like
        Dependent variable values (e.g., energy, polarization, bond order).
    x_series : pandas.Series or array-like
        Independent variable values (e.g., iteration index or time).
    mode : {'max', 'min', 'minmax'}, optional
        Type of extrema to extract:
        - ``'max'``: maxima only
        - ``'min'``: minima only
        - ``'minmax'``: both minima and maxima
    chunk_size : float or int, optional
        Size of the x-axis window used to find local extrema. If not provided,
        only global extrema are returned.

    Returns
    -------
    list of tuple[float, float]
        List of ``(x, y)`` pairs corresponding to detected extrema points.

    Examples
    --------
    >>> get_extrema_points(energy, iters, mode="min")
    >>> get_extrema_points(signal, time, mode="minmax", chunk_size=50)
    """

    assert mode in ['max', 'min', 'minmax'], "Mode must be 'max', 'min', or 'minmax'"
    assert len(y_series) == len(x_series), "Series must be of the same length"

    x_series = pd.Series(x_series).reset_index(drop=True)
    y_series = pd.Series(y_series).reset_index(drop=True)

    def _extreme_idx(chunk, kind):
        return chunk.idxmax() if kind == 'max' else chunk.idxmin()

    results = []

    if chunk_size:
        min_x, max_x = x_series.min(), x_series.max()
        bins = np.arange(min_x, max_x + chunk_size, chunk_size)

        for i in range(len(bins) - 1):
            x_start, x_end = bins[i], bins[i + 1]
            mask = (x_series >= x_start) & (x_series < x_end)
            if not mask.any():
                continue

            y_chunk = y_series[mask]

            if mode in ['max', 'minmax']:
                idx = _extreme_idx(y_chunk, 'max')
                results.append((x_series.loc[idx], y_series.loc[idx]))

            if mode in ['min', 'minmax']:
                idx = _extreme_idx(y_chunk, 'min')
                results.append((x_series.loc[idx], y_series.loc[idx]))

    else:
        if mode in ['max', 'minmax']:
            idx = _extreme_idx(y_series, 'max')
            results.append((x_series.loc[idx], y_series.loc[idx]))

        if mode in ['min', 'minmax']:
            idx = _extreme_idx(y_series, 'min')
            results.append((x_series.loc[idx], y_series.loc[idx]))

    return results
