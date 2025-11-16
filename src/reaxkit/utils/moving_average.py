"""Smoothing utilities providing simple and exponential moving averages for ReaxKit time-series data."""

from __future__ import annotations
from typing import Optional, Union
import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.Series, list, tuple]

def moving_average(
    y: ArrayLike,
    window: int = 5,
    *,
    center: bool = True,
    min_periods: Optional[int] = 1,
) -> pd.Series:
    """
    Centered simple moving average (SMA).
    Returns a pandas Series (keeps index if input was a Series).
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
    Exponential moving average (EMA).
    Provide either `alpha` in (0,1] or `window` (alpha = 2/(window+1)).
    Returns a pandas Series (keeps index if input was a Series).
    """
    if alpha is None:
        if window is None or window < 1:
            raise ValueError("Provide alpha in (0,1] or a window >= 1.")
        alpha = 2.0 / (window + 1.0)
    s = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y, dtype=float))
    return s.ewm(alpha=float(alpha), adjust=adjust).mean()
