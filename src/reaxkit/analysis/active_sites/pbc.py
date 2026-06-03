"""Provide periodic-cell helpers for active-site trajectory analyses.

This module implements cell-matrix conversion and minimum-image operations used
by active-site structural and event analyzers. It is scoped to geometric/PBC
utilities and does not execute analyzer task workflows directly.

**Usage context**

- Cell handling: Build Cartesian cell matrices from lengths/angles per frame.
- PBC distances: Evaluate minimum-image vectors and pair distances.
- Active-site tasks: Reuse shared periodic geometry logic across modules.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from reaxkit.domain.data_models import ConnectivityTrajectoryData, SimulationData, TrajectoryData


def cell_matrix_from_lengths_angles(lengths: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Build a 3x3 cell matrix from lengths and angles.

    Parameters
    -----
    lengths : np.ndarray
        Lattice lengths `[a, b, c]`.
    angles_deg : np.ndarray
        Lattice angles `[alpha, beta, gamma]` in degrees.

    Returns
    -----
    np.ndarray
        `3x3` Cartesian cell matrix with lattice vectors as rows.

    Examples
    -----
    ```python
    h = cell_matrix_from_lengths_angles(np.array([10.0, 10.0, 20.0]), np.array([90.0, 90.0, 120.0]))
    ```
    Sample output:
    `array([[...], [...], [...]])`
    Meaning:
    The matrix maps fractional vectors into Cartesian coordinates.
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
    """Return the frame-specific cell matrix when simulation metadata is present.

    Parameters
    -----
    data : ConnectivityTrajectoryData | TrajectoryData
        Input trajectory container with optional simulation cell arrays.
    frame_index : int
        Frame index for which the cell matrix is requested.

    Returns
    -----
    Optional[np.ndarray]
        `3x3` cell matrix for the frame, or `None` if unavailable/invalid.

    Examples
    -----
    ```python
    cell = frame_cell_matrix(data, 0)
    ```
    Sample output:
    `array([[...], [...], [...]])` or `None`
    Meaning:
    Active-site analyzers can branch between periodic and non-periodic logic.
    """
    if isinstance(data, ConnectivityTrajectoryData):
        cell = _cell_from_simulation(data.trajectory.simulation, frame_index)
        if cell is not None:
            return cell
        return _cell_from_simulation(data.connectivity.simulation, frame_index)
    return _cell_from_simulation(data.simulation, frame_index)


def minimum_image_vectors(delta: np.ndarray, cell: Optional[np.ndarray]) -> np.ndarray:
    """Apply minimum-image convention to one or more displacement vectors.

    Parameters
    -----
    delta : np.ndarray
        Displacement vectors, shape `(n, 3)` or broadcast-compatible.
    cell : Optional[np.ndarray]
        `3x3` cell matrix. If `None`, vectors are returned unchanged.

    Returns
    -----
    np.ndarray
        Minimum-image-adjusted displacement vectors in Cartesian space.

    Examples
    -----
    ```python
    adj = minimum_image_vectors(delta, cell)
    ```
    Sample output:
    `array([...])`
    Meaning:
    Distances computed from `adj` respect periodic wrapping.
    """
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
    """Apply minimum-image convention to a single displacement vector.

    Parameters
    -----
    delta : np.ndarray
        Single Cartesian displacement vector, shape `(3,)`.
    cell : Optional[np.ndarray]
        `3x3` cell matrix. If `None`, the input vector is returned.

    Returns
    -----
    np.ndarray
        Minimum-image-adjusted single displacement vector.

    Examples
    -----
    ```python
    v = minimum_image_vector(np.array([9.5, 0.0, 0.0]), cell)
    ```
    Sample output:
    `array([-0.5, 0.0, 0.0])`
    Meaning:
    The vector is wrapped to the shortest periodic image.
    """
    return minimum_image_vectors(np.asarray(delta, dtype=float).reshape(1, 3), cell)[0]


def pairwise_min_image_distances(a: np.ndarray, b: np.ndarray, cell: Optional[np.ndarray]) -> np.ndarray:
    """Compute pairwise distances between point sets with optional PBC.

    Parameters
    -----
    a : np.ndarray
        First point set, shape `(na, 3)`.
    b : np.ndarray
        Second point set, shape `(nb, 3)`.
    cell : Optional[np.ndarray]
        `3x3` cell matrix. If provided, minimum-image distances are used.

    Returns
    -----
    np.ndarray
        Distance matrix of shape `(na, nb)`.

    Examples
    -----
    ```python
    d = pairwise_min_image_distances(c_positions, o_positions, cell)
    ```
    Sample output:
    `array([[...], [...]])`
    Meaning:
    Each entry is the shortest periodic distance between one `a` point and one `b` point.
    """
    if a.size == 0 or b.size == 0:
        return np.empty((a.shape[0], b.shape[0]), dtype=float)
    delta = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    delta = minimum_image_vectors(delta.reshape(-1, 3), cell).reshape(delta.shape)
    return np.sqrt(np.sum(delta * delta, axis=2))
