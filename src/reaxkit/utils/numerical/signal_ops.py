"""
Signal-processing utilities for binary state detection.

This module provides helper functions for applying hysteresis-based
state detection and post-processing of boolean time-series data,
commonly encountered in ReaxFF simulations and field-driven analyses.

Typical use cases include:

- detecting ON/OFF states in polarization or dipole signals
- applying Schmitt-trigger hysteresis to noisy response curves
- removing spurious state flickering in thresholded time series
"""

from __future__ import annotations
import numpy as np
from typing import Optional

def schmitt_hysteresis(y: np.ndarray, th: float, hys: float, init_on: Optional[bool] = None) -> np.ndarray:
    """
    Apply Schmitt-trigger hysteresis to a 1D signal.

    This function converts a continuous signal into a boolean state
    using separate ON and OFF thresholds to suppress noise-induced
    state switching.

    The switching rules are:
    - ON when ``y >= th + hys / 2``
    - OFF when ``y <= th - hys / 2``

    Parameters
    ----------
    y : array-like
        Input signal values (e.g., polarization, dipole moment).
    th : float
        Central threshold value.
    hys : float
        Total hysteresis width.
    init_on : bool, optional
        Initial state at the first sample. If not provided, the
        initial state is inferred from the first data point.

    Returns
    -------
    numpy.ndarray
        Boolean array representing the ON/OFF state over the signal.

    Examples
    --------
    >>> state = schmitt_hysteresis(signal, th=0.0, hys=0.1)
    """
    y = np.asarray(y, dtype=float)
    h = max(0.0, float(hys))
    th_on  = float(th) + h/2.0
    th_off = float(th) - h/2.0

    state = np.zeros_like(y, dtype=bool)
    on = (bool(init_on) if init_on is not None else (y[0] >= th_on))
    for i, v in enumerate(y):
        if not on and v >= th_on:
            on = True
        elif on and v <= th_off:
            on = False
        state[i] = on
    return state

def clean_flicker(state: np.ndarray, min_run: int) -> np.ndarray:
    """
    Remove short-lived state transitions in a boolean sequence.

    This function suppresses brief ON or OFF segments shorter than a
    specified minimum run length, producing a cleaner and more
    physically meaningful state trajectory.

    Parameters
    ----------
    state : array-like of bool
        Boolean state sequence (e.g., output of ``schmitt_hysteresis``).
    min_run : int
        Minimum number of consecutive samples required to retain a
        state segment.

    Returns
    -------
    numpy.ndarray
        Cleaned boolean state array with flicker removed.

    Examples
    --------
    >>> clean_state = clean_flicker(state, min_run=5)
    """
    s = np.asarray(state, dtype=bool)
    if min_run <= 1 or s.size == 0:
        return s

    ints = s.astype(int)
    edges = np.diff(ints, prepend=ints[0])
    idx = np.flatnonzero(edges != 0)
    starts = np.r_[0, idx]
    ends   = np.r_[idx, len(s)]
    vals   = ints[starts]

    keep = np.ones_like(vals, dtype=bool)
    for i in range(len(vals)):
        if (ends[i] - starts[i]) < min_run and len(vals) > 1:
            keep[i] = False

    out = np.empty_like(ints)
    cursor = 0
    last_val = None
    i = 0
    while i < len(vals):
        if not keep[i]:
            j = i + 1
            while j < len(vals) and not keep[j]:
                j += 1
            val = vals[i-1] if i > 0 else (vals[j] if j < len(vals) else vals[i])
            end = ends[j-1] if j > 0 else ends[i]
            out[cursor:end] = val
            cursor = end
            i = j
            continue
        j = i + 1
        end = ends[i]
        while j < len(vals) and not keep[j]:
            end = ends[j]
            j += 1
        out[cursor:end] = vals[i]
        cursor = end
        last_val = vals[i]
        i = j

    if cursor < len(out):
        out[cursor:] = last_val if last_val is not None else ints[0]
    return out.astype(bool)
