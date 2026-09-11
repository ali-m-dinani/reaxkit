"""Shared geometry helpers for z-binned stress/strain analyzers.

This module centralizes trajectory validation, frame and atom selection, cell
matrix construction, and periodic coordinate unwrapping used by the z-binned
strain tasks. It contains no CLI or presentation behavior.

**Usage context**

- Analyzer reuse: Keep both z-binned strain methods on identical selections.
- Periodic systems: Build triclinic cell matrices and unwrap selected atoms.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import numpy as np

from reaxkit.domain.data_models import TrajectoryData


def selected_frames(n_frames: int, frames: Iterable[int] | None, every: int) -> list[int]:
    """Return validated zero-based frame indices in chronological order."""
    if every < 1:
        raise ValueError("every must be at least 1.")
    values = list(range(n_frames)) if frames is None else [int(frame) for frame in frames]
    if not values:
        raise ValueError("At least one frame must be selected.")
    invalid = [frame for frame in values if frame < 0 or frame >= n_frames]
    if invalid:
        raise IndexError(f"Frame indices outside [0, {n_frames - 1}]: {invalid}")
    if len(set(values)) != len(values):
        raise ValueError("Frame selection contains duplicate frame indices.")
    return sorted(values)[::every]


def select_atom_indices(elements: Sequence[str], atom_types: Sequence[str] | None) -> np.ndarray:
    """Return atom-column indices matching element names case-insensitively."""
    if atom_types is None or len(atom_types) == 0:
        return np.arange(len(elements), dtype=int)
    requested = {str(value).strip().casefold() for value in atom_types if str(value).strip()}
    if not requested:
        raise ValueError("atom_types must contain at least one non-empty element name.")
    indices = np.asarray(
        [index for index, element in enumerate(elements) if str(element).strip().casefold() in requested],
        dtype=int,
    )
    if indices.size == 0:
        available = sorted({str(element) for element in elements})
        raise ValueError(f"No atoms match {sorted(requested)}. Available elements: {available}")
    return indices


def bin_edges(z_coordinates: np.ndarray, z_bins: int) -> np.ndarray:
    """Return equal-width edges spanning finite z coordinates."""
    if z_bins < 1:
        raise ValueError("z_bins must be at least 1.")
    values = np.asarray(z_coordinates, dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("Bin coordinates must be a non-empty finite one-dimensional array.")
    lower, upper = float(np.min(values)), float(np.max(values))
    if np.isclose(lower, upper, rtol=0.0, atol=1.0e-12):
        raise ValueError("Selected atoms have no z extent, so z bins cannot be constructed.")
    return np.linspace(lower, upper, z_bins + 1, dtype=float)


def assign_bins(z_coordinates: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign one-based bins, placing interior-edge values in the upper bin."""
    zero_based = np.searchsorted(edges, z_coordinates, side="right") - 1
    return np.clip(zero_based, 0, len(edges) - 2).astype(int) + 1


def trajectory_cell_matrices(data: TrajectoryData) -> np.ndarray | None:
    """Build one row-vector cell matrix per trajectory frame when available."""
    simulation = data.simulation
    if simulation is None or simulation.cell_lengths is None:
        return None
    lengths = np.asarray(simulation.cell_lengths, dtype=float)
    if lengths.shape != (data.positions.shape[0], 3):
        raise ValueError("simulation.cell_lengths must have shape (n_frames, 3).")
    angles = (
        np.full_like(lengths, 90.0)
        if simulation.cell_angles is None
        else np.asarray(simulation.cell_angles, dtype=float)
    )
    if angles.shape != lengths.shape:
        raise ValueError("simulation.cell_angles must have shape (n_frames, 3).")
    return np.stack([_cell_matrix(length, angle) for length, angle in zip(lengths, angles)])


def _cell_matrix(lengths: Sequence[float], angles_deg: Sequence[float]) -> np.ndarray:
    """Construct a row-vector triclinic cell matrix."""
    a, b, c = np.asarray(lengths, dtype=float)
    alpha, beta, gamma = np.radians(np.asarray(angles_deg, dtype=float))
    if np.any(~np.isfinite([a, b, c, alpha, beta, gamma])) or min(a, b, c) <= 0.0:
        raise ValueError("Cell lengths and angles must be finite and lengths must be positive.")
    sin_gamma = float(np.sin(gamma))
    if abs(sin_gamma) < 1.0e-12:
        raise ValueError("The gamma cell angle produces a singular cell.")
    cos_alpha, cos_beta, cos_gamma = np.cos(alpha), np.cos(beta), np.cos(gamma)
    vector_c_x = c * cos_beta
    vector_c_y = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    vector_c_z_sq = c * c - vector_c_x * vector_c_x - vector_c_y * vector_c_y
    if vector_c_z_sq < -1.0e-10:
        raise ValueError("Cell lengths and angles do not define a valid cell.")
    matrix = np.asarray(
        [
            [a, 0.0, 0.0],
            [b * cos_gamma, b * sin_gamma, 0.0],
            [vector_c_x, vector_c_y, np.sqrt(max(0.0, vector_c_z_sq))],
        ],
        dtype=float,
    )
    if abs(float(np.linalg.det(matrix))) < 1.0e-12:
        raise ValueError("Cell matrix is singular.")
    return matrix


def periodic_axes(value: str | Sequence[bool]) -> tuple[bool, bool, bool]:
    """Normalize an axes string or three booleans to periodic-axis flags."""
    if isinstance(value, str):
        axes = value.strip().lower()
        if axes == "none":
            return False, False, False
        invalid = sorted(set(axes).difference("xyz"))
        if invalid:
            raise ValueError(f"periodic contains invalid axes: {invalid}")
        return tuple(axis in axes for axis in "xyz")  # type: ignore[return-value]
    flags = tuple(bool(item) for item in value)
    if len(flags) != 3:
        raise ValueError("periodic must be an axes string or three Boolean values.")
    return flags  # type: ignore[return-value]


def iter_selected_coordinates(
    data: TrajectoryData,
    eligible_indices: np.ndarray,
    output_frames: Sequence[int],
    *,
    unwrap: bool,
    periodic: str | Sequence[bool],
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield selected-atom coordinates and finite masks chronologically."""
    positions = np.asarray(data.positions, dtype=float)
    cells = trajectory_cell_matrices(data)
    axes = periodic_axes(periodic)
    if unwrap and cells is None:
        raise ValueError("Periodic unwrapping requires simulation cell lengths.")

    wanted = set(int(frame) for frame in output_frames)
    reference = positions[0, eligible_indices]
    tracking_valid = np.all(np.isfinite(reference), axis=1)
    previous_fractional = reference @ np.linalg.inv(cells[0]) if unwrap else None
    unwrapped_fractional = previous_fractional.copy() if unwrap else None

    for frame_index in range(max(output_frames) + 1):
        wrapped = positions[frame_index, eligible_indices]
        wrapped_finite = np.all(np.isfinite(wrapped), axis=1)
        if frame_index == 0:
            coordinates = reference.copy()
        elif unwrap:
            assert cells is not None and previous_fractional is not None and unwrapped_fractional is not None
            current_fractional = np.full_like(wrapped, np.nan, dtype=float)
            current_fractional[wrapped_finite] = wrapped[wrapped_finite] @ np.linalg.inv(cells[frame_index])
            step_valid = tracking_valid & wrapped_finite
            step = current_fractional[step_valid] - previous_fractional[step_valid]
            for axis, is_periodic in enumerate(axes):
                if is_periodic:
                    step[:, axis] -= np.round(step[:, axis])
            unwrapped_fractional[step_valid] += step
            tracking_valid &= wrapped_finite
            coordinates = np.full_like(wrapped, np.nan, dtype=float)
            coordinates[tracking_valid] = unwrapped_fractional[tracking_valid] @ cells[frame_index]
            previous_fractional[wrapped_finite] = current_fractional[wrapped_finite]
        else:
            coordinates = wrapped
            tracking_valid = wrapped_finite
        if frame_index in wanted:
            yield frame_index, coordinates, np.all(np.isfinite(coordinates), axis=1)


def iteration_values(data: TrajectoryData) -> np.ndarray:
    """Return one iteration value per loaded frame."""
    if data.iterations is not None:
        return np.asarray(data.iterations, dtype=int)
    if data.simulation is not None and data.simulation.iterations is not None:
        return np.asarray(data.simulation.iterations, dtype=int)
    return np.arange(data.positions.shape[0], dtype=int)
