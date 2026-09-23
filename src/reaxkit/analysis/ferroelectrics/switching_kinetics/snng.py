"""Site-saturated nucleation with nucleation and growth (SNNG)."""

from __future__ import annotations

import numpy as np
from scipy.special import gamma, gammainc


def snng_growth_integral(time, alpha: float, m: float) -> np.ndarray:
    r"""Evaluate ``integral g'(tau) (t-tau)^2 d tau`` analytically.

    Here ``g'(tau) = alpha*m*tau**(m-1)*exp(-alpha*tau**m)``.  Expanding
    ``(t-tau)^2`` reduces the convolution to three lower incomplete gamma
    moments, avoiding a nested numerical integral during optimization.
    """
    values = np.asarray(time, dtype=float)
    if alpha <= 0.0 or m <= 0.0:
        raise ValueError("SNNG parameters alpha and m must be greater than zero.")
    if np.any(values < 0.0):
        raise ValueError("Switching times must be non-negative.")

    z = float(alpha) * np.power(values, float(m))
    moments: list[np.ndarray] = []
    for order in (0, 1, 2):
        shape = 1.0 + order / float(m)
        moment = (
            float(alpha) ** (-order / float(m))
            * gamma(shape)
            * gammainc(shape, z)
        )
        moments.append(np.asarray(moment, dtype=float))
    integral = values**2 * moments[0] - 2.0 * values * moments[1] + moments[2]
    return np.maximum(integral, 0.0)


def snng_fraction(time, prefactor: float, alpha: float, m: float) -> np.ndarray:
    r"""Return ``1-exp(-prefactor * integral)`` for the SNNG model.

    The identifiable prefactor is ``2*pi*d*v**2*N_infinity``.  Thickness
    ``d``, wall velocity ``v``, and saturated nucleation density
    ``N_infinity`` cannot be separated using a switched-fraction curve alone.
    """
    if prefactor <= 0.0:
        raise ValueError("The SNNG prefactor must be greater than zero.")
    integral = snng_growth_integral(time, alpha, m)
    return -np.expm1(-float(prefactor) * integral)


def snng_fraction_physical(
    time,
    thickness: float,
    wall_velocity: float,
    nucleation_density: float,
    alpha: float,
    m: float,
) -> np.ndarray:
    """Evaluate SNNG from physical factors when all three are known."""
    if thickness <= 0.0 or wall_velocity <= 0.0 or nucleation_density <= 0.0:
        raise ValueError("Thickness, wall velocity, and nucleation density must be positive.")
    prefactor = 2.0 * np.pi * thickness * wall_velocity**2 * nucleation_density
    return snng_fraction(time, prefactor, alpha, m)


__all__ = ["snng_fraction", "snng_fraction_physical", "snng_growth_integral"]
