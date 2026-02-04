"""
Numerical analysis helper utilities.

This module provides lightweight numerical tools for analyzing one-dimensional
data series, such as detecting zero crossings or sign changes in curves
produced by ReaxFF simulations.

Typical use cases include:

- locating x-axis crossings in energy or force curves
- identifying switching points in polarization or field-response signals
- detecting sign changes in derived observables
"""


from __future__ import annotations
from typing import Sequence, List
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq


def find_zero_crossings(x: Sequence[float], y: Sequence[float]) -> List[float]:
    """
    Find x-values where a 1D function crosses zero.

    Zero crossings are identified by detecting sign changes between consecutive
    data points and estimating the root location using interpolation and
    numerical bracketing. Exact zeros at sampled points are also included.

    Parameters
    ----------
    x : sequence of float
        Monotonically ordered x-values (e.g., time, iteration index).
    y : sequence of float
        Function values corresponding to ``x``.

    Returns
    -------
    list of float
        Sorted x-values at which ``y(x) = 0``.

    Examples
    --------
    >>> find_zero_crossings(time, polarization)
    >>> find_zero_crossings(iters, energy - energy.mean())
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have the same length")

    # exact zeros at sample points
    zeros = list(x_arr[y_arr == 0])

    if x_arr.size < 2:
        return sorted(set(zeros))

    interp = interp1d(x_arr, y_arr, fill_value="extrapolate")

    for i in range(len(y_arr) - 1):
        if np.isnan(y_arr[i]) or np.isnan(y_arr[i+1]):
            continue
        if y_arr[i] * y_arr[i+1] < 0:
            try:
                root = brentq(interp, x_arr[i], x_arr[i+1])
                zeros.append(root)
            except ValueError:
                # If brentq fails for some weird numerical reason, skip that interval
                pass

    sorted(set(zeros))
    zeros = [float(z) for z in zeros]
    return zeros
