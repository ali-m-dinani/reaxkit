"""Signal-processing helpers (i.e., applying Schmitt-trigger hysteresis) in time-series data."""

from __future__ import annotations
import numpy as np
from typing import Optional

def schmitt_hysteresis(y: np.ndarray, th: float, hys: float, init_on: Optional[bool] = None) -> np.ndarray:
    """
    Boolean state with Schmitt-trigger hysteresis.
      - ON when y >= th + hys/2
      - OFF when y <= th - hys/2
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
    Remove very short on/off segments (length < min_run) in a boolean sequence.
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
