"""Expose reusable numerical utility operations for ReaxKit.

This package provides reusable numerical operations such as filtering, moving
averages, extrema detection, and signal processing primitives. It serves as a
convenience import surface for common numerical utilities used in workflows.

**Usage context**

- Package imports: Access commonly used numerical helpers from one namespace.
- Analysis workflows: Reuse shared signal and spectrum computation primitives.
- Utility composition: Build higher-level analysis steps from stable helpers.
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
