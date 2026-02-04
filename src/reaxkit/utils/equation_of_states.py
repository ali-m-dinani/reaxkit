"""
Equation of state (EOS) utilities.

This module centralizes common equation-of-state formulas used across ReaxKit.
Currently includes Vinet EOS in two forms:

1) Energy–volume form in eV (used for fitting E(V) from fort.99/fort.74).
2) Legacy "trainset" form matching the translated Fortran elastic_energy_v2 generator.

Notes
-----
- The two Vinet implementations use different parameterizations/units, so they are
  *conceptually related* but not drop-in replacements for each other.
- Explanation for the Rose–Vinet equation of state can be found here:
    doi:10.1029/JB092iB09p09319
"""

from __future__ import annotations

from typing import Union
import numpy as np


def vinet_energy_ev(V: np.ndarray, E0: float, K0_eV_A3: float, V0: float, C: float) -> np.ndarray:
    """
    Vinet EOS: energy–volume form (eV, eV/Å^3).

    Parameters
    ----------
    V : numpy.ndarray
        Volume(s) (Å^3).
    E0 : float
        Equilibrium energy (eV).
    K0_eV_A3 : float
        Bulk modulus at equilibrium (eV/Å^3).
    V0 : float
        Equilibrium volume (Å^3).
    C : float
        Vinet shape parameter (dimensionless).

    Returns
    -------
    numpy.ndarray
        Energy at each volume V (eV).
    """
    nu = V / V0
    eta = nu ** (1.0 / 3.0)
    term = 1.0 - (1.0 + C * (eta - 1.0)) * np.exp(C * (1.0 - eta))
    return E0 + 9.0 * K0_eV_A3 * V0 / (C ** 2) * term


def vinet_energy_trainset(
    *,
    volume: float,
    reference_volume: float,
    bulk_modulus_gpa: float,
    bulk_modulus_pressure_derivative: float,
    reference_energy: float = 0.0,
    energy_conversion_factor: float,
) -> float:
    """
    Vinet EOS: legacy trainset-generator form.

    This matches your translated Fortran elastic_energy_v2 bulk block logic.
    It returns energies in the generator's legacy energy units.

    Parameters
    ----------
    volume : float
        Current volume V (Å^3).
    reference_volume : float
        Reference volume V0 (Å^3).
    bulk_modulus_gpa : float
        Bulk modulus B0 (GPa).
    bulk_modulus_pressure_derivative : float
        Pressure derivative B0' (dimensionless).
    reference_energy : float, optional
        Reference energy offset E0 (legacy units).
    energy_conversion_factor : float
        The generator's conversion factor (the one you currently call ENERGY_CONVERSION_FACTOR).

    Returns
    -------
    float
        EOS energy in the generator's legacy energy units.
    """
    converted_bulk_modulus = bulk_modulus_gpa / energy_conversion_factor
    eos_prefactor = 9.0 * converted_bulk_modulus * reference_volume / 16.0

    x = (reference_volume / volume) ** (2.0 / 3.0)
    x_minus_one = x - 1.0

    term_cubic = (x_minus_one ** 3) * bulk_modulus_pressure_derivative
    term_quadratic = (x_minus_one ** 2) * (6.0 - 4.0 * x)

    return reference_energy + eos_prefactor * (term_cubic + term_quadratic)
