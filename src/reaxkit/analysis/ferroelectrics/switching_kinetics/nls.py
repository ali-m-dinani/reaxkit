"""Nucleation-limited-switching (NLS) kinetics."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8)
def _unit_interval_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights mapped to the open unit interval."""
    if order < 16:
        raise ValueError("NLS quadrature order must be at least 16.")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def nls_fraction(
    time,
    t1: float,
    width: float,
    n: float,
    amplitude: float = 1.0,
    *,
    quadrature_order: int = 96,
) -> np.ndarray:
    r"""Return the Lorentzian-distributed NLS switched fraction.

    The model is

    ``A integral [1-exp(-(t/t0)**n)] F(log(t0)) d(log(t0))``,

    where ``F`` is a normalized Lorentzian centered on ``log(t1)`` with
    half-width ``width``.  The Cauchy inverse-CDF transform converts the
    infinite integral to a stable Gauss-Legendre integral on ``(0, 1)``.
    Natural logarithms are used; changing log base only rescales ``width``.
    """
    values = np.asarray(time, dtype=float)
    if t1 <= 0.0 or width <= 0.0 or n <= 0.0 or amplitude <= 0.0:
        raise ValueError("NLS parameters t1, width, n, and amplitude must be positive.")
    if np.any(values < 0.0):
        raise ValueError("Switching times must be non-negative.")

    probabilities, weights = _unit_interval_quadrature(int(quadrature_order))
    log_t0 = np.log(float(t1)) + float(width) * np.tan(
        np.pi * (probabilities - 0.5)
    )
    # Evaluate (t/t0)^n in log space and clip before exponentiation.  Values
    # beyond this range already produce a KAI fraction indistinguishable from
    # zero or one in double precision.
    positive_time = values > 0.0
    switched = np.zeros_like(values, dtype=float)
    if np.any(positive_time):
        log_ratio_power = float(n) * (
            np.log(values[positive_time])[:, None] - log_t0[None, :]
        )
        ratio_power = np.exp(np.clip(log_ratio_power, -745.0, 40.0))
        kernels = -np.expm1(-ratio_power)
        switched[positive_time] = kernels @ weights
    return float(amplitude) * switched


__all__ = ["nls_fraction"]
