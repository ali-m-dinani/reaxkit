"""Periodic-boundary helpers for trajectory analyses."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from reaxkit.domain.data_models import TrajectoryData


def _cell_matrix(lengths: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    a, b, c = [float(v) for v in lengths]
    alpha, beta, gamma = np.deg2rad(np.asarray(angles_deg, dtype=float))

    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sg = np.sin(gamma)
    if not np.isfinite(sg) or abs(sg) < 1e-12:
        raise ValueError("Invalid gamma angle for cell matrix construction.")

    ax = np.array([a, 0.0, 0.0], dtype=float)
    bx = np.array([b * cg, b * sg, 0.0], dtype=float)
    cx = np.array(
        [
            c * cb,
            c * (ca - cb * cg) / sg,
            c * np.sqrt(max(0.0, 1.0 - cb * cb - ((ca - cb * cg) / sg) ** 2)),
        ],
        dtype=float,
    )
    return np.vstack([ax, bx, cx])


def _unwrap_single_atom(coords: np.ndarray, cells: np.ndarray, angles: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    out = np.zeros_like(coords, dtype=float)
    out[0] = coords[0]

    h_prev = _cell_matrix(cells[0], angles[0])
    h_prev_inv = np.linalg.inv(h_prev)
    f_prev = coords[0] @ h_prev_inv

    for i in range(1, n):
        h_curr = _cell_matrix(cells[i], angles[i])
        h_curr_inv = np.linalg.inv(h_curr)
        f_curr = coords[i] @ h_curr_inv
        d_frac = f_curr - f_prev
        d_frac -= np.round(d_frac)
        out[i] = out[i - 1] + (d_frac @ h_prev)
        h_prev = h_curr
        h_prev_inv = h_curr_inv
        f_prev = f_curr
    return out


def maybe_unwrap_selected_positions(
    data: TrajectoryData,
    *,
    frame_idx: Sequence[int],
    sel_idx: Sequence[int],
    unwrap: bool,
) -> np.ndarray:
    """Return selected coordinates; unwrap across PBC when possible."""
    fi = np.asarray([int(v) for v in frame_idx], dtype=int)
    si = np.asarray([int(v) for v in sel_idx], dtype=int)
    coords = np.asarray(data.positions[fi][:, si, :], dtype=float)
    if not unwrap:
        return coords

    sim = data.simulation
    if sim is None or sim.cell_lengths is None:
        return coords

    cell_l = np.asarray(sim.cell_lengths, dtype=float)
    if cell_l.ndim != 2 or cell_l.shape[1] != 3 or cell_l.shape[0] <= np.max(fi):
        return coords
    cell_l = cell_l[fi]

    if sim.cell_angles is not None:
        cell_a = np.asarray(sim.cell_angles, dtype=float)
        if cell_a.ndim != 2 or cell_a.shape[1] != 3 or cell_a.shape[0] <= np.max(fi):
            return coords
        cell_a = cell_a[fi]
    else:
        cell_a = np.full_like(cell_l, 90.0, dtype=float)

    out = np.asarray(coords, dtype=float).copy()
    n_atoms = out.shape[1]
    for j in range(n_atoms):
        atom_coords = out[:, j, :]
        if not np.isfinite(atom_coords).all():
            continue
        out[:, j, :] = _unwrap_single_atom(atom_coords, cell_l, cell_a)
    return out

