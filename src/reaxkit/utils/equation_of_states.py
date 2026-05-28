"""
Equation of state (EOS) utilities.

This module centralizes common equation-of-state formulas used across ReaxKit.
It is responsible for reusable EOS computations and is not tied to a single
workflow implementation.

**Usage context**

- fitting energy-volume relationships from parsed simulation outputs
- computing legacy trainset-style energies for translated generator logic

Notes
-----
- The two Vinet implementations use different parameterizations/units, so they
  are conceptually related but not drop-in replacements for each other.
- Explanation for the Rose-Vinet equation of state can be found here:
    doi:10.1029/JB092iB09p09319
"""

from __future__ import annotations

import numpy as np


def vinet_energy_ev(V: np.ndarray, E0: float, K0_eV_A3: float, V0: float, C: float) -> np.ndarray:
    """
    Compute Vinet EOS energy from volume in eV units.

    This evaluates the energy-volume form used for fitting E(V) data in
    ReaxKit workflows, with bulk modulus expressed in eV/Angstrom^3.

    Parameters
    ----------
    V : numpy.ndarray
        Volume(s) (Angstrom^3).
    E0 : float
        Equilibrium energy (eV).
    K0_eV_A3 : float
        Bulk modulus at equilibrium (eV/Angstrom^3).
    V0 : float
        Equilibrium volume (Angstrom^3).
    C : float
        Vinet shape parameter (dimensionless).

    Returns
    -------
    numpy.ndarray
        Energy at each volume `V` (eV).

    Examples
    --------
    >>> import numpy as np
    >>> V = np.array([15.0, 16.0, 17.0])
    >>> E = vinet_energy_ev(V, E0=-10.2, K0_eV_A3=0.75, V0=16.0, C=4.0)
    >>> E.shape
    (3,)
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
    Compute legacy trainset-style EOS energy from scalar volume input.

    This matches the translated Fortran `elastic_energy_v2` bulk-block logic.
    It returns energies in the generator's legacy energy units and preserves
    the existing conversion convention through `energy_conversion_factor`.

    Parameters
    ----------
    volume : float
        Current volume `V` (Angstrom^3).
    reference_volume : float
        Reference volume `V0` (Angstrom^3).
    bulk_modulus_gpa : float
        Bulk modulus `B0` (GPa).
    bulk_modulus_pressure_derivative : float
        Pressure derivative `B0'` (dimensionless).
    reference_energy : float, optional
        Reference energy offset `E0` (legacy units).
    energy_conversion_factor : float
        Generator conversion factor currently used in trainset workflows.

    Returns
    -------
    float
        EOS energy in the generator's legacy energy units.

    Examples
    --------
    >>> e = vinet_energy_trainset(
    ...     volume=16.0,
    ...     reference_volume=16.2,
    ...     bulk_modulus_gpa=180.0,
    ...     bulk_modulus_pressure_derivative=4.2,
    ...     reference_energy=0.0,
    ...     energy_conversion_factor=160.21766208,
    ... )
    >>> isinstance(e, float)
    True
    """
    converted_bulk_modulus = bulk_modulus_gpa / energy_conversion_factor
    eos_prefactor = 9.0 * converted_bulk_modulus * reference_volume / 16.0

    x = (reference_volume / volume) ** (2.0 / 3.0)
    x_minus_one = x - 1.0

    term_cubic = (x_minus_one ** 3) * bulk_modulus_pressure_derivative
    term_quadratic = (x_minus_one ** 2) * (6.0 - 4.0 * x)

    return reference_energy + eos_prefactor * (term_cubic + term_quadratic)
