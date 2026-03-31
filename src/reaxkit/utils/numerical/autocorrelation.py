"""
Autocorrelation utility helpers.

This module provides reusable autocorrelation functions for one-dimensional
signals used in analysis tasks and workflows.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float], list, tuple]
NormalizeMode = Literal["none", "biased", "unbiased", "coeff"]


def _as_1d_float(signal: ArrayLike) -> np.ndarray:
    arr = np.asarray(signal, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"signal must be 1D; got shape={arr.shape}")
    if arr.size < 2:
        raise ValueError("signal must contain at least two samples.")
    if not np.isfinite(arr).all():
        raise ValueError("signal contains non-finite values.")
    return arr


def autocorrelation(
    signal: ArrayLike,
    *,
    max_lag: Optional[int] = None,
    normalize: NormalizeMode = "coeff",
    center: bool = True,
) -> dict[str, np.ndarray]:
    """
    Compute autocorrelation for non-negative lags.

    Parameters
    ----------
    signal : array-like
        Input 1D signal.
    max_lag : int, optional
        Maximum lag to return. Default returns all lags ``0..N-1``.
    normalize : {'none', 'biased', 'unbiased', 'coeff'}, optional
        Normalization mode:
        - 'none': raw correlation sum
        - 'biased': divide by N
        - 'unbiased': divide by N-lag
        - 'coeff': normalized so lag-0 equals 1.0
    center : bool, optional
        If True, subtract signal mean before correlation.

    Returns
    -------
    dict[str, numpy.ndarray]
        Dictionary with keys:
        - 'lag': integer lag indices
        - 'acf': autocorrelation values for each lag
    """
    y = _as_1d_float(signal)
    if center:
        y = y - float(np.mean(y))

    n = y.size
    if max_lag is None:
        max_lag_i = n - 1
    else:
        max_lag_i = int(max_lag)
        if max_lag_i < 0:
            raise ValueError("max_lag must be >= 0.")
        max_lag_i = min(max_lag_i, n - 1)

    full = np.correlate(y, y, mode="full")
    acf = full[n - 1 : n + max_lag_i].astype(float)
    lag = np.arange(max_lag_i + 1, dtype=int)

    mode = str(normalize).strip().lower()
    if mode == "none":
        pass
    elif mode == "biased":
        acf = acf / float(n)
    elif mode == "unbiased":
        acf = acf / (n - lag).astype(float)
    elif mode == "coeff":
        denom = float(acf[0])
        if np.isclose(denom, 0.0):
            acf = np.zeros_like(acf, dtype=float)
        else:
            acf = acf / denom
    else:
        raise ValueError("normalize must be one of: 'none', 'biased', 'unbiased', 'coeff'.")

    return {"lag": lag, "acf": acf}


def autocorrelation_time(
    signal: ArrayLike,
    *,
    dt: float = 1.0,
    max_lag: Optional[int] = None,
    normalize: NormalizeMode = "coeff",
    center: bool = True,
) -> dict[str, np.ndarray]:
    """
    Compute autocorrelation vs physical time lag.

    Returns
    -------
    dict[str, numpy.ndarray]
        Dictionary with keys:
        - 'lag': lag index
        - 'tau': lag time (lag * dt)
        - 'acf': autocorrelation values
    """
    if float(dt) <= 0.0:
        raise ValueError("dt must be > 0.")
    out = autocorrelation(signal, max_lag=max_lag, normalize=normalize, center=center)
    lag = out["lag"]
    tau = lag.astype(float) * float(dt)
    return {"lag": lag, "tau": tau, "acf": out["acf"]}


__all__ = [
    "NormalizeMode",
    "autocorrelation",
    "autocorrelation_time",
]

