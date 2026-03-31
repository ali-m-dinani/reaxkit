"""
Numerical utility helpers.

This package provides reusable numerical operations such as filtering, moving
averages, extrema detection, and signal processing primitives.
"""

from reaxkit.utils.numerical.fft import (
    DetrendMode,
    WindowName,
    detrend_signal,
    dominant_frequency,
    fft_spectrum,
    window_function,
)
from reaxkit.utils.numerical.autocorrelation import (
    NormalizeMode,
    autocorrelation,
    autocorrelation_time,
)

__all__ = [
    "DetrendMode",
    "WindowName",
    "NormalizeMode",
    "detrend_signal",
    "window_function",
    "fft_spectrum",
    "dominant_frequency",
    "autocorrelation",
    "autocorrelation_time",
]
