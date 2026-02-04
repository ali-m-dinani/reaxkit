"""
Time-series smoothing utilities.

This module provides simple and exponential moving-average functions for
smoothing one-dimensional data series commonly produced by ReaxFF simulations,
such as energies, bond orders, dipole moments, or polarization signals.

Typical use cases include:

- reducing high-frequency noise in MD trajectories
- smoothing field-response or hysteresis curves
- preparing time-series data for extrema or trend analysis
"""


from __future__ import annotations
from typing import Optional, Union
import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.Series, list, tuple]

def simple_moving_average(
    y: ArrayLike,
    window: int = 5,
    *,
    center: bool = True,
    min_periods: Optional[int] = 1,
) -> pd.Series:
    """
    Compute a simple moving average (SMA) of a 1D data series.

    The moving average is computed over a fixed-size sliding window and
    returned as a pandas Series. If the input is already a Series, its index
    is preserved.

    Parameters
    ----------
    y : array-like
        Input data values to smooth.
    window : int, optional
        Size of the moving window.
    center : bool, optional
        Whether the window is centered on each data point.
    min_periods : int, optional
        Minimum number of observations required to compute a value.

    Returns
    -------
    pandas.Series
        Smoothed data series using a simple moving average.

    Examples
    --------
    >>> simple_moving_average(energy, window=10)
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    s = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y, dtype=float))
    return s.rolling(window, center=center, min_periods=min_periods).mean()

def exponential_moving_average(
    y: ArrayLike,
    *,
    window: Optional[int] = None,
    alpha: Optional[float] = None,
    adjust: bool = False,
) -> pd.Series:
    """
    Compute an exponential moving average (EMA) of a 1D data series.

    The exponential moving average applies exponentially decreasing weights
    to past observations. The smoothing factor may be specified directly
    via ``alpha`` or indirectly via a window size.

    Parameters
    ----------
    y : array-like
        Input data values to smooth.
    window : int, optional
        Window size used to derive the smoothing factor
        (``alpha = 2 / (window + 1)``).
    alpha : float, optional
        Smoothing factor in the interval ``(0, 1]``.
    adjust : bool, optional
        Whether to use bias-adjusted weights.

    Returns
    -------
    pandas.Series
        Smoothed data series using an exponential moving average.

    Examples
    --------
    >>> exponential_moving_average(signal, window=8)
    >>> exponential_moving_average(signal, alpha=0.2)
    """
    if alpha is None:
        if window is None or window < 1:
            raise ValueError("Provide alpha in (0,1] or a window >= 1.")
        alpha = 2.0 / (window + 1.0)
    s = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y, dtype=float))
    return s.ewm(alpha=float(alpha), adjust=adjust).mean()
