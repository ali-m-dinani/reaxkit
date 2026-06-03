"""
Fourier-transform utility helpers.

This module provides reusable FFT helpers for one-dimensional signals used by
analysis tasks and workflows. It handles common preprocessing choices such as
detrending and windowing, and returns frequency-domain arrays ready for
plotting or downstream feature extraction.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float], list, tuple]
DetrendMode = Literal["none", "mean", "linear"]
WindowName = Literal["none", "hann", "hamming", "blackman", "bartlett"]


def _as_1d_float(signal: ArrayLike) -> np.ndarray:
    """Validate and convert input data to a finite 1D float array."""
    arr = np.asarray(signal, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"signal must be 1D; got shape={arr.shape}")
    if arr.size < 2:
        raise ValueError("signal must contain at least two samples for FFT.")
    if not np.isfinite(arr).all():
        raise ValueError("signal contains non-finite values.")
    return arr


def detrend_signal(signal: ArrayLike, mode: DetrendMode = "mean") -> np.ndarray:
    """Remove low-order trends from a 1D signal.

    Applies one of the supported detrending modes to prepare signal data for
    frequency-domain analysis.

    Parameters
    -----
    signal : array-like
        Input 1D signal.
    mode : {'none', 'mean', 'linear'}, optional
        Detrending mode:
        - 'none': no detrending
        - 'mean': remove mean value
        - 'linear': remove best-fit linear trend

    Returns
    -----
    numpy.ndarray
        Detrended signal.
    Examples
    -----
    >>> detrend_signal([1.0, 2.0, 3.0], mode="mean")
    array([-1.,  0.,  1.])
    """
    y = _as_1d_float(signal).copy()
    mode_l = str(mode).strip().lower()
    if mode_l == "none":
        return y
    if mode_l == "mean":
        return y - float(np.mean(y))
    if mode_l == "linear":
        x = np.arange(y.size, dtype=float)
        m, b = np.polyfit(x, y, 1)
        return y - (m * x + b)
    raise ValueError("mode must be one of: 'none', 'mean', 'linear'.")


def window_function(
    n: int,
    name: WindowName = "hann",
    *,
    kaiser_beta: Optional[float] = None,
) -> np.ndarray:
    """Build a 1D window function.

    Constructs standard tapering windows used before FFT computation and also
    supports an optional Kaiser window path.

    Parameters
    -----
    n : int
        Number of samples.
    name : {'none', 'hann', 'hamming', 'blackman', 'bartlett'}, optional
        Window type.
    kaiser_beta : float, optional
        If provided and ``name == 'none'``, uses a Kaiser window with this beta.
        This allows optional tunable windowing without expanding the enum.

    Returns
    -----
    numpy.ndarray
        Window values with shape (n,).
    Examples
    -----
    >>> window_function(8, name="hann")
    array([...])
    """
    if n < 1:
        raise ValueError("n must be >= 1.")
    name_l = str(name).strip().lower()
    if kaiser_beta is not None and name_l == "none":
        return np.kaiser(int(n), float(kaiser_beta)).astype(float)
    if name_l == "none":
        return np.ones(int(n), dtype=float)
    if name_l == "hann":
        return np.hanning(int(n)).astype(float)
    if name_l == "hamming":
        return np.hamming(int(n)).astype(float)
    if name_l == "blackman":
        return np.blackman(int(n)).astype(float)
    if name_l == "bartlett":
        return np.bartlett(int(n)).astype(float)
    raise ValueError("name must be one of: 'none', 'hann', 'hamming', 'blackman', 'bartlett'.")


def fft_spectrum(
    signal: ArrayLike,
    dt: float,
    *,
    detrend: DetrendMode = "mean",
    window: WindowName = "hann",
    kaiser_beta: Optional[float] = None,
    one_sided: bool = True,
) -> dict[str, np.ndarray]:
    """Compute FFT spectrum for a uniformly sampled 1D signal.

    Performs detrending, applies the selected window, computes FFT values, and
    returns common spectrum components for downstream analysis and plotting.

    Parameters
    -----
    signal : array-like
        Input time-domain signal.
    dt : float
        Sampling interval in time units.
    detrend : {'none', 'mean', 'linear'}, optional
        Detrending mode before FFT.
    window : {'none', 'hann', 'hamming', 'blackman', 'bartlett'}, optional
        Window function to apply before FFT.
    kaiser_beta : float, optional
        Optional Kaiser beta parameter (used when ``window='none'``).
    one_sided : bool, optional
        If True, return non-negative frequency part only.

    Returns
    -----
    dict[str, numpy.ndarray]
        Dictionary with keys:
        - 'freq': frequency axis
        - 'amplitude': amplitude spectrum
        - 'power': power spectrum (amplitude squared)
        - 'phase': phase angle (radians)
        - 'real': real FFT component
        - 'imag': imaginary FFT component

    Notes
    -----
    - For help, see numpy's documentation on `numpy.fft` and `scipy.signal` for windowing and detrending, which can be found at https://numpy.org/doc/2.2/reference/generated/numpy.fft.rfft.html
    - An example can be found at https://hackmd.io/@cccccccc/S1DE042eO

    Examples
    -----
    >>> fft_spectrum([0.0, 1.0, 0.0, -1.0], dt=0.1)
    {'freq': array([...]), 'amplitude': array([...]), ...}
    """
    if float(dt) <= 0.0:
        raise ValueError("dt must be > 0.")

    y = detrend_signal(signal, mode=detrend)
    n = y.size
    win = window_function(n, name=window, kaiser_beta=kaiser_beta)
    y_win = y * win

    if one_sided:
        fft_vals = np.fft.rfft(y_win)
        freq = np.fft.rfftfreq(n, d=float(dt))
    else:
        fft_vals = np.fft.fft(y_win)
        freq = np.fft.fftfreq(n, d=float(dt))

    amp = np.abs(fft_vals) / float(n)
    if one_sided and n > 1:
        if n % 2 == 0:
            amp[1:-1] *= 2.0
        else:
            amp[1:] *= 2.0
    power = amp * amp

    return {
        "freq": freq.astype(float),
        "amplitude": amp.astype(float),
        "power": power.astype(float),
        "phase": np.angle(fft_vals).astype(float),
        "real": np.real(fft_vals).astype(float),
        "imag": np.imag(fft_vals).astype(float),
    }


def dominant_frequency(
    signal: ArrayLike,
    dt: float,
    *,
    detrend: DetrendMode = "mean",
    window: WindowName = "hann",
    kaiser_beta: Optional[float] = None,
    min_freq: float = 0.0,
) -> float:
    """Return dominant positive frequency of a 1D signal.

    Computes a one-sided spectrum and selects the frequency with maximum
    amplitude among frequencies that satisfy `freq >= min_freq`.

    Parameters
    -----
    signal : array-like
        Input time-domain signal.
    dt : float
        Sampling interval in time units.
    detrend : {'none', 'mean', 'linear'}, optional
        Detrending mode before FFT.
    window : {'none', 'hann', 'hamming', 'blackman', 'bartlett'}, optional
        Window function to apply before FFT.
    kaiser_beta : float, optional
        Optional Kaiser beta parameter (used when ``window='none'``).
    min_freq : float, optional
        Minimum frequency threshold for candidate dominant frequency.

    Returns
    -----
    float
        Dominant frequency value.

    Examples
    -----
    >>> dominant_frequency([0.0, 1.0, 0.0, -1.0], dt=0.1)
    2.5
    """
    spec = fft_spectrum(
        signal,
        dt,
        detrend=detrend,
        window=window,
        kaiser_beta=kaiser_beta,
        one_sided=True,
    )
    f = spec["freq"]
    a = spec["amplitude"]
    mask = f >= float(min_freq)
    if not np.any(mask):
        raise ValueError("No frequencies satisfy min_freq.")
    f_sel = f[mask]
    a_sel = a[mask]
    if f_sel.size == 0:
        raise ValueError("No frequency points available after filtering.")
    return float(f_sel[int(np.argmax(a_sel))])


__all__ = [
    "DetrendMode",
    "WindowName",
    "detrend_signal",
    "window_function",
    "fft_spectrum",
    "dominant_frequency",
]
