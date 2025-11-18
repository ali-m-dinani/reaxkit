# reaxkit/utils/numerical_analysis_utils.py

from __future__ import annotations
from typing import Sequence, List
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq


def _find_zero_crossings(x: Sequence[float], y: Sequence[float]) -> List[float]:
    """
    Find approximate x-positions where y(x) = 0 using linear interpolation
    and bracketing between consecutive points.

    Returns a sorted list of unique roots.
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
