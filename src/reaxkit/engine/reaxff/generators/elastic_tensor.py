"""Cartesian small-strain conventions shared by elastic targets and geometries."""

from __future__ import annotations

import numpy as np


LEGACY_MODES = ("c11", "c22", "c33", "c12", "c13", "c23", "c44", "c55", "c66")
TENSOR_MODES = LEGACY_MODES + tuple(
    f"c{first}{second}"
    for first in range(1, 7)
    for second in range(first + 1, 7)
    if f"c{first}{second}" not in LEGACY_MODES
)


def validate_tensor(tensor) -> np.ndarray:
    """Require a finite symmetric stiffness matrix in engineering Voigt notation."""
    matrix = np.asarray(tensor, dtype=float)
    if matrix.shape != (6, 6) or not np.isfinite(matrix).all():
        raise ValueError("tensor_gpa must be a finite 6x6 matrix.")
    if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-6):
        raise ValueError("tensor_gpa must be symmetric (Cij = Cji).")
    return (matrix + matrix.T) / 2


def validate_cell(cell) -> None:
    """Reject nonphysical lattice metrics before generating tensor targets."""
    lengths = np.array([cell.a, cell.b, cell.c], dtype=float)
    angles = np.array([cell.alpha, cell.beta, cell.gamma], dtype=float)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("Cell lengths must be finite and positive.")
    if not np.isfinite(angles).all() or np.any(angles <= 0) or np.any(angles >= 180):
        raise ValueError("Cell angles must be finite and between 0 and 180 degrees.")
    cosine_alpha, cosine_beta, cosine_gamma = np.cos(np.radians(angles))
    metric = np.array([[1, cosine_gamma, cosine_beta], [cosine_gamma, 1, cosine_alpha], [cosine_beta, cosine_alpha, 1]])
    if np.linalg.eigvalsh(metric).min() <= 1e-12:
        raise ValueError("Cell angles must define a non-degenerate positive-definite lattice metric.")


def mode_strain(mode: str) -> np.ndarray:
    """Return d(eta)/d(delta); eta uses xx, yy, zz, 2yz, 2xz, 2xy."""
    if mode not in TENSOR_MODES:
        raise ValueError(f"Unknown elastic mode: {mode!r}")
    first, second = int(mode[1]) - 1, int(mode[2]) - 1
    strain = np.zeros(6)
    strain[first] = 1.0 if first < 3 else 2.0
    if second != first:
        strain[second] = -1.0 if second < 3 else 2.0
    return strain


def strain_matrix(voigt) -> np.ndarray:
    """Convert engineering Voigt strain into a symmetric Cartesian matrix."""
    normal_x, normal_y, normal_z, shear_yz, shear_xz, shear_xy = voigt
    return np.array([
        [normal_x, shear_xy / 2, shear_xz / 2],
        [shear_xy / 2, normal_y, shear_yz / 2],
        [shear_xz / 2, shear_yz / 2, normal_z],
    ])


def tensor_warnings(tensor) -> list[str]:
    """Report instability without rejecting intentionally unstable reference phases."""
    matrix = validate_tensor(tensor)
    minimum = float(np.linalg.eigvalsh(matrix).min())
    if minimum <= 0:
        return [f"Elastic tensor is not positive definite (minimum eigenvalue {minimum:.6g} GPa); reference phase is mechanically unstable or marginal."]
    return []
