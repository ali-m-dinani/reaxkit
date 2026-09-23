"""Kolmogorov-Avrami-Ishibashi (KAI) switching kinetics."""

from __future__ import annotations

import numpy as np

KAI_CHARACTERISTIC_FRACTION = 1.0 - np.exp(-1.0)


def kai_fraction(time, t0: float, n: float) -> np.ndarray:
    r"""Return the switched fraction ``1 - exp(-(t / t0)**n)``.

    ``t0`` is the characteristic switching time and ``n`` is the Avrami
    exponent.  At ``t == t0`` the switched fraction is ``1 - exp(-1)``.
    """
    values = np.asarray(time, dtype=float)
    if t0 <= 0.0 or n <= 0.0:
        raise ValueError("KAI parameters t0 and n must be greater than zero.")
    if np.any(values < 0.0):
        raise ValueError("Switching times must be non-negative.")
    exponent = np.power(values / float(t0), float(n))
    return -np.expm1(-exponent)


def estimate_kai_t0(time, fraction) -> float:
    """Interpolate the time at which the switched fraction reaches ``1-e^-1``.

    A cumulative maximum suppresses small downward noise excursions without
    otherwise smoothing the observations. The time array must already use the
    desired zero-time origin and units.
    """
    values = np.asarray(time, dtype=float)
    switched = np.asarray(fraction, dtype=float)
    if values.ndim != 1 or switched.ndim != 1 or values.size != switched.size:
        raise ValueError("Time and switched fraction must be equal-length 1D arrays.")
    if values.size < 2 or not np.all(np.isfinite(values)) or not np.all(np.isfinite(switched)):
        raise ValueError("At least two finite time/fraction samples are required to estimate t0.")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("Time values must be strictly increasing to estimate t0.")
    monotonic_fraction = np.maximum.accumulate(switched)
    index = int(np.searchsorted(monotonic_fraction, KAI_CHARACTERISTIC_FRACTION))
    if index == 0:
        raise ValueError(
            "The first sample is already at or above the KAI characteristic fraction; "
            "an earlier sample is required to estimate t0."
        )
    if index >= values.size:
        raise ValueError(
            "The data never reach the KAI characteristic fraction "
            f"{KAI_CHARACTERISTIC_FRACTION:.6f}; t0 cannot be fixed automatically."
        )
    lower_fraction = float(monotonic_fraction[index - 1])
    upper_fraction = float(monotonic_fraction[index])
    weight = (KAI_CHARACTERISTIC_FRACTION - lower_fraction) / (
        upper_fraction - lower_fraction
    )
    crossing = float(values[index - 1] + weight * (values[index] - values[index - 1]))
    if crossing <= 0.0 or not np.isfinite(crossing):
        raise ValueError("The interpolated KAI t0 must be finite and greater than zero.")
    return crossing


__all__ = ["KAI_CHARACTERISTIC_FRACTION", "estimate_kai_t0", "kai_fraction"]
