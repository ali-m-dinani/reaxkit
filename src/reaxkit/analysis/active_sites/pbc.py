"""Periodic-cell helpers for active-site analysis."""

from __future__ import annotations

from typing import Optional

import numpy as np

from reaxkit.domain.data_models import ConnectivityTrajectoryData, SimulationData, TrajectoryData


def cell_matrix_from_lengths_angles(lengths: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Build a 3x3 cell matrix from lengths and angles.

    Parameters
    ----------
    lengths
        Array-like [a, b, c].
    angles_deg
        Array-like [alpha, beta, gamma] in degrees.
    """
    a, b, c = [float(v) for v in np.asarray(lengths, dtype=float).reshape(3)]
    alpha, beta, gamma = np.radians(np.asarray(angles_deg, dtype=float).reshape(3))

    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sg = np.sin(gamma)
    if abs(sg) < 1.0e-12:
        return np.zeros((3, 3), dtype=float)

    v1 = np.array([a, 0.0, 0.0], dtype=float)
    v2 = np.array([b * cg, b * sg, 0.0], dtype=float)
    cx = c * cb
    cy = c * (ca - cb * cg) / sg
    cz_sq = c * c - cx * cx - cy * cy
    cz = np.sqrt(max(0.0, float(cz_sq)))
    v3 = np.array([cx, cy, cz], dtype=float)
    return np.vstack([v1, v2, v3])


def _cell_from_simulation(simulation: Optional[SimulationData], frame_index: int) -> Optional[np.ndarray]:
    if simulation is None or simulation.cell_lengths is None:
        return None

    lengths = np.asarray(simulation.cell_lengths, dtype=float)
    if lengths.ndim != 2 or lengths.shape[1] != 3 or frame_index >= lengths.shape[0]:
        return None
    cur_lengths = lengths[frame_index]

    if simulation.cell_angles is not None:
        angles = np.asarray(simulation.cell_angles, dtype=float)
        if angles.ndim == 2 and angles.shape[1] == 3 and frame_index < angles.shape[0]:
            cur_angles = angles[frame_index]
        else:
            cur_angles = np.array([90.0, 90.0, 90.0], dtype=float)
    else:
        cur_angles = np.array([90.0, 90.0, 90.0], dtype=float)

    if not np.isfinite(cur_lengths).all() or not np.isfinite(cur_angles).all():
        return None
    return cell_matrix_from_lengths_angles(cur_lengths, cur_angles)


def frame_cell_matrix(data: ConnectivityTrajectoryData | TrajectoryData, frame_index: int) -> Optional[np.ndarray]:
    """Return frame cell matrix when available, else None."""
    if isinstance(data, ConnectivityTrajectoryData):
        cell = _cell_from_simulation(data.trajectory.simulation, frame_index)
        if cell is not None:
            return cell
        return _cell_from_simulation(data.connectivity.simulation, frame_index)
    return _cell_from_simulation(data.simulation, frame_index)


def minimum_image_vectors(delta: np.ndarray, cell: Optional[np.ndarray]) -> np.ndarray:
    """Apply minimum-image convention to displacement vectors."""
    if cell is None:
        return np.asarray(delta, dtype=float)
    delta = np.asarray(delta, dtype=float)
    try:
        inv = np.linalg.inv(cell)
    except np.linalg.LinAlgError:
        return delta
    frac = delta @ inv.T
    frac -= np.round(frac)
    return frac @ cell.T


def minimum_image_vector(delta: np.ndarray, cell: Optional[np.ndarray]) -> np.ndarray:
    """Apply minimum-image convention to one vector."""
    return minimum_image_vectors(np.asarray(delta, dtype=float).reshape(1, 3), cell)[0]


def pairwise_min_image_distances(a: np.ndarray, b: np.ndarray, cell: Optional[np.ndarray]) -> np.ndarray:
    """Pairwise distances between points in `a` and `b` with optional PBC."""
    if a.size == 0 or b.size == 0:
        return np.empty((a.shape[0], b.shape[0]), dtype=float)
    delta = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    delta = minimum_image_vectors(delta.reshape(-1, 3), cell).reshape(delta.shape)
    return np.sqrt(np.sum(delta * delta, axis=2))
