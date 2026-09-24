"""ReaxFF shielded Coulomb kernel and its analytic derivative."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

COULOMB_CONSTANT_KCAL_ANGSTROM_PER_MOL_E2 = 332.0638
KCAL_PER_MOL_PER_EV = 23.06054783061903
V_PER_ANGSTROM_TO_MV_PER_CM = 100.0


def build_taper_coefficients(lower: float, upper: float) -> tuple[float, ...]:
    lower, upper = float(lower), float(upper)
    if not np.isfinite([lower, upper]).all() or lower < 0.0 or upper <= lower:
        raise ValueError("Taper radii must be finite with 0 <= lower < upper.")
    d7 = (upper - lower) ** 7
    a2, a3, b2, b3 = lower**2, lower**3, upper**2, upper**3
    return (
        (-35*a3*b2*b2 + 21*a2*b3*b2 + 7*lower*b3*b3 + b3*b3*upper) / d7,
        140*a3*b3 / d7,
        -210*(a3*b2 + a2*b3) / d7,
        140*(a3*upper + 3*a2*b2 + lower*b3) / d7,
        -35*(a3 + 9*a2*upper + 9*lower*b2 + b3) / d7,
        84*(a2 + 3*lower*upper + b2) / d7,
        -70*(lower + upper) / d7,
        20 / d7,
    )


def evaluate_taper(distance, coefficients: Sequence[float]):
    values = np.asarray(distance, dtype=float)
    result = np.zeros_like(values)
    for coefficient in reversed(tuple(coefficients)):
        result = result * values + float(coefficient)
    return result


def evaluate_taper_derivative(distance, coefficients: Sequence[float]):
    values = np.asarray(distance, dtype=float)
    result = np.zeros_like(values)
    for power in range(7, 0, -1):
        result = result * values + power * float(coefficients[power])
    return result


def shielded_kernel(distance, gamma_i, gamma_j, taper=1.0):
    r = np.asarray(distance, dtype=float)
    gi, gj = np.asarray(gamma_i, dtype=float), np.asarray(gamma_j, dtype=float)
    base = r**3 + (gi * gj) ** -1.5
    return COULOMB_CONSTANT_KCAL_ANGSTROM_PER_MOL_E2 * np.asarray(taper) * base ** (-1/3)


def shielded_kernel_radial_derivative(distance, gamma_i, gamma_j, taper, taper_derivative):
    r = np.asarray(distance, dtype=float)
    base = r**3 + (np.asarray(gamma_i) * np.asarray(gamma_j)) ** -1.5
    return COULOMB_CONSTANT_KCAL_ANGSTROM_PER_MOL_E2 * (
        np.asarray(taper_derivative) * base ** (-1/3)
        - np.asarray(taper) * r**2 * base ** (-4/3)
    )
