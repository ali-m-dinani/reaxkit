"""analyzer to extract the minimum or maximum of a series of data"""
import numpy as np
import pandas as pd


def get_extrema_points(y_series, x_series, mode='max', chunk_size=None):
    """Find extrema points (x, y) where a y-series reaches its max/min.

    Parameters
    ----------
    y_series : pd.Series
        Target data (e.g., electric field or energy).
    x_series : pd.Series
        Corresponding x values (e.g., iteration numbers or time).
    mode : str
        'max', 'min', or 'minmax' to find global/local peaks and/or valleys.
    chunk_size : float or int, optional
        If given, defines x-axis window size to find local extrema (e.g., 75 units in time).

    Returns
    -------
    List[Tuple[float, float]]
        List of (x, y) pairs where extrema occur.
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
