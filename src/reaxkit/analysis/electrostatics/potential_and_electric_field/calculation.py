"""Per-frame ReaxFF Coulomb energy, potential, and local electric field."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np

from .parameters import ReaxFFCoulombParameters
from .physics import (
    KCAL_PER_MOL_PER_EV,
    evaluate_taper,
    evaluate_taper_derivative,
    shielded_kernel,
    shielded_kernel_radial_derivative,
)

MIN_DISTANCE = 0.001
TARGET_BATCH_SIZE = 1024
MAX_ALL_PAIRS_PER_BATCH = 1_000_000


@dataclass
class FrameElectrostatics:
    per_atom_coulomb_kcal_per_mol: np.ndarray
    total_coulomb_kcal_per_mol: float
    probe_labels: tuple[str, ...]
    internal_probe_potential_v: np.ndarray
    internal_probe_field_v_per_angstrom: np.ndarray
    distinct_interactions: int
    self_image_interactions: int


def reaxff_cell_matrix(lengths: Sequence[float], angles_degrees: Sequence[float]) -> np.ndarray:
    """Build the row-vector cell orientation used by standalone ReaxFF."""
    a, b, c = np.asarray(lengths, dtype=float)
    alpha, beta, gamma = np.deg2rad(np.asarray(angles_degrees, dtype=float))
    sa, ca, sb, cb = np.sin(alpha), np.cos(alpha), np.sin(beta), np.cos(beta)
    cos_phi = (np.cos(gamma) - ca * cb) / (sa * sb)
    cos_phi = float(np.clip(cos_phi, -1.0, 1.0))
    cell = np.asarray([
        [a * sb * np.sqrt(max(0.0, 1.0 - cos_phi**2)), a * sb * cos_phi, a * cb],
        [0.0, b * sa, b * ca],
        [0.0, 0.0, c],
    ])
    if not np.isfinite(cell).all() or abs(float(np.linalg.det(cell))) < 1e-12:
        raise ValueError("Cell lengths and angles define a singular ReaxFF cell.")
    return cell


def _translations(cell: np.ndarray, cutoff: float, periodic: tuple[bool, bool, bool]) -> tuple[np.ndarray, np.ndarray]:
    inverse = np.linalg.inv(cell)
    bounds = [int(np.floor(cutoff * np.linalg.norm(inverse[:, axis]) + 1.0)) if flag else 0
              for axis, flag in enumerate(periodic)]
    indices = np.asarray(list(product(*(range(-bound, bound + 1) for bound in bounds))), dtype=int)
    if len(indices) > 1_000_000:
        raise ValueError("Periodic image enumeration exceeds 1,000,000 translations.")
    return indices, indices @ cell


def _canonical_self_indices(indices: np.ndarray) -> np.ndarray:
    keep = []
    for row in indices:
        first = next((int(value) for value in row if value != 0), 0)
        keep.append(first > 0)
    return indices[np.asarray(keep, dtype=bool)]


def _wrap(positions: np.ndarray, cell: np.ndarray, periodic: tuple[bool, bool, bool]) -> np.ndarray:
    fractional = positions @ np.linalg.inv(cell)
    for axis, enabled in enumerate(periodic):
        if enabled:
            fractional[:, axis] -= np.floor(fractional[:, axis])
    return fractional @ cell


def _minimum_image(vectors: np.ndarray, cell: np.ndarray | None, periodic: tuple[bool, bool, bool]) -> np.ndarray:
    if cell is None or not any(periodic):
        return vectors
    fractional = vectors @ np.linalg.inv(cell)
    for axis, enabled in enumerate(periodic):
        if enabled:
            fractional[..., axis] -= np.rint(fractional[..., axis])
    return fractional @ cell


def _kernel_gradient(vectors: np.ndarray, gamma_i, gamma_j, parameters, method: str,
                     step: float, disable_taper: bool) -> np.ndarray:
    if method == "numerical":
        gradient = np.empty_like(vectors)
        for axis in range(3):
            plus, minus = vectors.copy(), vectors.copy()
            plus[:, axis] += step
            minus[:, axis] -= step
            gradient[:, axis] = (
                _masked_kernel(plus, gamma_i, gamma_j, parameters, disable_taper)
                - _masked_kernel(minus, gamma_i, gamma_j, parameters, disable_taper)
            ) / (2.0 * step)
        return gradient
    distances = np.linalg.norm(vectors, axis=1)
    accepted = distances >= MIN_DISTANCE
    if not disable_taper:
        accepted &= distances <= parameters.upper_taper_radius
    gradient = np.zeros_like(vectors)
    if not np.any(accepted):
        return gradient
    r = distances[accepted]
    gi = np.broadcast_to(np.asarray(gamma_i), distances.shape)[accepted]
    gj = np.broadcast_to(np.asarray(gamma_j), distances.shape)[accepted]
    taper = np.ones_like(r) if disable_taper else evaluate_taper(r, parameters.taper_coefficients)
    dtaper = np.zeros_like(r) if disable_taper else evaluate_taper_derivative(r, parameters.taper_coefficients)
    radial = shielded_kernel_radial_derivative(r, gi, gj, taper, dtaper)
    gradient[accepted] = radial[:, None] * vectors[accepted] / r[:, None]
    return gradient


def _probe_kernels_and_radial_factors(vectors: np.ndarray, probe_gammas: np.ndarray,
                                      source_gammas: np.ndarray, parameters,
                                      disable_taper: bool) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate all analytic probe kernels with one distance/taper calculation."""
    distances = np.linalg.norm(vectors, axis=1)
    accepted = distances >= MIN_DISTANCE
    if not disable_taper:
        accepted &= distances <= parameters.upper_taper_radius
    kernels = np.zeros((len(probe_gammas), len(distances)), dtype=float)
    radial_factors = np.zeros_like(kernels)
    if not np.any(accepted):
        return kernels, radial_factors
    r = distances[accepted]
    gi = np.asarray(probe_gammas, dtype=float)[:, None]
    gj = np.asarray(source_gammas, dtype=float)[accepted][None, :]
    taper = np.ones_like(r) if disable_taper else evaluate_taper(r, parameters.taper_coefficients)
    dtaper = np.zeros_like(r) if disable_taper else evaluate_taper_derivative(r, parameters.taper_coefficients)
    kernels[:, accepted] = shielded_kernel(r[None, :], gi, gj, taper[None, :])
    radial = shielded_kernel_radial_derivative(
        r[None, :], gi, gj, taper[None, :], dtaper[None, :],
    )
    radial_factors[:, accepted] = radial / r[None, :]
    return kernels, radial_factors


def _masked_kernel(vectors: np.ndarray, gamma_i, gamma_j, parameters, disable_taper: bool) -> np.ndarray:
    distances = np.linalg.norm(vectors, axis=1)
    accepted = distances >= MIN_DISTANCE
    if not disable_taper:
        accepted &= distances <= parameters.upper_taper_radius
    output = np.zeros_like(distances)
    if np.any(accepted):
        r = distances[accepted]
        gi = np.broadcast_to(np.asarray(gamma_i), distances.shape)[accepted]
        gj = np.broadcast_to(np.asarray(gamma_j), distances.shape)[accepted]
        taper = np.ones_like(r) if disable_taper else evaluate_taper(r, parameters.taper_coefficients)
        output[accepted] = shielded_kernel(r, gi, gj, taper)
    return output


def _tree_pairs(tree, xyz: np.ndarray, targets: np.ndarray, translation: np.ndarray,
                cutoff: float, *, translation_sign: float) -> tuple[np.ndarray, np.ndarray]:
    """Flatten neighbor lists into aligned target/source index arrays."""
    neighborhoods = tree.query_ball_point(
        xyz[targets] + translation_sign * translation, cutoff,
        workers=-1 if len(xyz) >= 1000 else 1,
    )
    counts = np.fromiter((len(values) for values in neighborhoods), dtype=np.int64,
                         count=len(neighborhoods))
    if not np.any(counts):
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    return np.repeat(targets, counts), np.concatenate(neighborhoods).astype(int, copy=False)


def _all_pairs(targets: np.ndarray, atom_count: int) -> tuple[np.ndarray, np.ndarray]:
    sources = np.tile(np.arange(atom_count, dtype=int), len(targets))
    return np.repeat(targets, atom_count), sources


def _self_image_kernel_sums(vectors: np.ndarray, gammas: np.ndarray,
                            parameters) -> tuple[np.ndarray, int]:
    distances = np.linalg.norm(vectors, axis=1)
    accepted = ((distances >= MIN_DISTANCE)
                & (distances <= parameters.upper_taper_radius))
    if not np.any(accepted):
        return np.zeros(len(gammas), dtype=float), 0
    r = distances[accepted]
    taper = evaluate_taper(r, parameters.taper_coefficients)
    gamma = np.asarray(gammas, dtype=float)[:, None]
    kernels = shielded_kernel(r[None, :], gamma, gamma, taper[None, :])
    return kernels.sum(axis=1), int(len(gammas) * np.count_nonzero(accepted))


def calculate_frame(
    positions: np.ndarray,
    charges: np.ndarray,
    labels: Sequence[str],
    parameters: ReaxFFCoulombParameters,
    *,
    probe_elements: Sequence[str],
    cell: np.ndarray | None = None,
    periodic: Sequence[bool] = (False, False, False),
    field_method: str = "analytic",
    field_step: float = 0.001,
    disable_taper: bool = False,
) -> FrameElectrostatics:
    """Evaluate local observables; the target atom is excluded at its own location."""
    xyz, q = np.asarray(positions, dtype=float), np.asarray(charges, dtype=float)
    periodic = tuple(bool(value) for value in periodic)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or q.shape != (len(xyz),):
        raise ValueError("positions and charges must have shapes (atoms, 3) and (atoms,).")
    if len(labels) != len(xyz) or not np.isfinite(xyz).all() or not np.isfinite(q).all():
        raise ValueError("Atom labels and finite coordinates/charges are required for every atom.")
    if field_method not in {"analytic", "numerical"}:
        raise ValueError("field_method must be analytic or numerical.")
    if field_method == "numerical" and (not np.isfinite(field_step) or field_step <= 0):
        raise ValueError("field_step must be finite and positive.")
    if any(periodic) and cell is None:
        raise ValueError("Periodic calculations require a cell.")
    if cell is not None:
        cell = np.asarray(cell, dtype=float)
        xyz = _wrap(xyz, cell, periodic)
    if disable_taper:
        indices, translations = np.zeros((1, 3), dtype=int), np.zeros((1, 3))
    elif cell is not None and any(periodic):
        indices, translations = _translations(cell, parameters.upper_taper_radius, periodic)
    else:
        indices, translations = np.zeros((1, 3), dtype=int), np.zeros((1, 3))
    gammas = parameters.gamma_values(labels)
    per_atom = np.zeros(len(xyz))
    pair_count = 0
    if not disable_taper:
        from scipy.spatial import cKDTree
        tree = cKDTree(xyz)
        for translation in translations:
            for start in range(0, len(xyz), TARGET_BATCH_SIZE):
                targets = np.arange(start, min(start + TARGET_BATCH_SIZE, len(xyz)))
                pair_targets, sources = _tree_pairs(
                    tree, xyz, targets, translation, parameters.upper_taper_radius,
                    translation_sign=-1.0,
                )
                keep = sources < pair_targets
                pair_targets, sources = pair_targets[keep], sources[keep]
                if not len(sources):
                    continue
                vectors = xyz[sources] - xyz[pair_targets] + translation
                kernels = _masked_kernel(vectors, gammas[sources], gammas[pair_targets], parameters, False)
                energies = q[sources] * q[pair_targets] * kernels
                per_atom += 0.5 * np.bincount(sources, weights=energies, minlength=len(xyz))
                per_atom += 0.5 * np.bincount(pair_targets, weights=energies, minlength=len(xyz))
                pair_count += int(np.count_nonzero(kernels))
    else:
        for i in range(max(0, len(xyz) - 1)):
            js = np.arange(i + 1, len(xyz))
            vectors = _minimum_image(xyz[i] - xyz[js], cell, periodic)
            kernels = _masked_kernel(vectors, gammas[i], gammas[js], parameters, True)
            energies = q[i] * q[js] * kernels
            per_atom[i] += 0.5 * energies.sum()
            per_atom[js] += 0.5 * energies
            pair_count += int(np.count_nonzero(kernels))
    self_count = 0
    if cell is not None and any(periodic) and not disable_taper:
        self_indices = _canonical_self_indices(indices)
        self_vectors = self_indices @ cell
        kernel_sums, self_count = _self_image_kernel_sums(self_vectors, gammas, parameters)
        per_atom += q ** 2 * kernel_sums

    probe_labels = tuple(str(value) for value in probe_elements)
    probe_gammas = parameters.gamma_values(probe_labels)
    potential = np.zeros((len(probe_labels), len(xyz)))
    field = np.zeros((len(probe_labels), len(xyz), 3))
    all_atoms = np.arange(len(xyz))
    tree = None
    if not disable_taper:
        from scipy.spatial import cKDTree
        tree = cKDTree(xyz)
    for image_index, translation in enumerate(translations):
        zero = bool(np.all(indices[image_index] == 0))
        batch_size = (TARGET_BATCH_SIZE if tree is not None else
                      max(1, MAX_ALL_PAIRS_PER_BATCH // max(1, len(xyz))))
        for start in range(0, len(xyz), batch_size):
            targets = all_atoms[start:min(start + batch_size, len(xyz))]
            if tree is not None:
                pair_targets, sources = _tree_pairs(
                    tree, xyz, targets, translation, parameters.upper_taper_radius,
                    translation_sign=1.0,
                )
            else:
                pair_targets, sources = _all_pairs(targets, len(xyz))
            if zero:
                keep = sources != pair_targets
                pair_targets, sources = pair_targets[keep], sources[keep]
            if not len(sources):
                continue
            vectors = xyz[pair_targets] - xyz[sources] + translation
            if disable_taper:
                vectors = _minimum_image(vectors, cell, periodic)
            analytic_kernels = analytic_radial = None
            if field_method == "analytic":
                analytic_kernels, analytic_radial = _probe_kernels_and_radial_factors(
                    vectors, probe_gammas, gammas[sources], parameters, disable_taper,
                )
            for probe_index, probe_gamma in enumerate(probe_gammas):
                if analytic_kernels is None:
                    kernels = _masked_kernel(vectors, probe_gamma, gammas[sources], parameters, disable_taper)
                    gradient = _kernel_gradient(vectors, probe_gamma, gammas[sources], parameters,
                                                field_method, field_step, disable_taper)
                    radial_factor = None
                else:
                    kernels = analytic_kernels[probe_index]
                    radial_factor = analytic_radial[probe_index]
                weighted_kernel = q[sources] * kernels
                potential[probe_index] += np.bincount(
                    pair_targets, weights=weighted_kernel, minlength=len(xyz),
                )
                for axis in range(3):
                    field_weights = (q[sources] * gradient[:, axis] if radial_factor is None else
                                     q[sources] * radial_factor * vectors[:, axis])
                    field[probe_index, :, axis] -= np.bincount(
                        pair_targets, weights=field_weights, minlength=len(xyz),
                    )
    potential /= KCAL_PER_MOL_PER_EV
    field /= KCAL_PER_MOL_PER_EV
    return FrameElectrostatics(per_atom, float(per_atom.sum()), probe_labels, potential, field,
                               pair_count, self_count)
