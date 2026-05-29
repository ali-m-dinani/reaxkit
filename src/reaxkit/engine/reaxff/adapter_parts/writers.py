"""Write-side formatting helpers for the ReaxFF adapter.

This module contains the data-shaping logic used by adapter write methods so
the adapter class can stay focused on API routing.

**Usage context**

- Control export: Convert `ControlParametersData` into ReaxFF control files.
- Trajectory export: Normalize `TrajectoryData` into xmolout frame records.
- Adapter internals: Called from `ReaxFFAdapter.write_*` methods.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reaxkit.domain.data_models import ControlParametersData, TrajectoryData


def _write_control_data(
    data: ControlParametersData,
    out_path: str | Path,
    args: dict | None = None,
):
    """Write ReaxFF control output from `ControlParametersData`."""
    from reaxkit.engine.reaxff.generators.control_generator import _write_control_from_data

    args = args or {}
    if not isinstance(data, ControlParametersData):
        raise TypeError("write_control expects ControlParametersData.")

    overrides = args.get("overrides") or args.get("control_overrides")
    return _write_control_from_data(
        data,
        out_path=out_path,
        overrides=overrides,
    )


def _write_trajectory_data(
    data: TrajectoryData,
    out_path: str | Path,
    args: dict | None = None,
):
    """Write xmolout output from `TrajectoryData` frames and metadata."""
    from reaxkit.engine.reaxff.generators.xmolout_generator import _write_xmolout_from_frames

    args = args or {}
    positions = np.asarray(data.positions, dtype=float)
    if positions.ndim != 3:
        raise ValueError("TrajectoryData.positions must have shape (n_frames, n_atoms, 3).")

    n_frames, n_atoms, _ = positions.shape
    if data.atom_labels is not None:
        atom_labels = np.asarray(data.atom_labels, dtype=object)
        if atom_labels.shape != (n_frames, n_atoms):
            raise ValueError("TrajectoryData.atom_labels must have shape (n_frames, n_atoms).")
    else:
        atom_labels = np.tile(np.asarray(data.elements, dtype=object), (n_frames, 1))

    iterations = (
        np.asarray(data.iterations, dtype=int)
        if data.iterations is not None
        else (
            np.asarray(data.simulation.iterations, dtype=int)
            if data.simulation is not None and data.simulation.iterations is not None
            else np.arange(n_frames, dtype=int)
        )
    )
    potential_energy = (
        np.asarray(data.simulation.potential_energy, dtype=float)
        if data.simulation is not None and data.simulation.potential_energy is not None
        else np.zeros((n_frames,), dtype=float)
    )
    cell_lengths = (
        np.asarray(data.simulation.cell_lengths, dtype=float)
        if data.simulation is not None and data.simulation.cell_lengths is not None
        else np.ones((n_frames, 3), dtype=float)
    )
    cell_angles = (
        np.asarray(data.simulation.cell_angles, dtype=float)
        if data.simulation is not None and data.simulation.cell_angles is not None
        else np.full((n_frames, 3), 90.0, dtype=float)
    )

    frames = []
    for fi in range(n_frames):
        frames.append(
            {
                "iter": int(iterations[fi]) if fi < len(iterations) else int(fi),
                "coords": positions[fi],
                "atom_types": [str(label) for label in atom_labels[fi].tolist()],
                "E_pot": float(potential_energy[fi]) if fi < len(potential_energy) else 0.0,
                "a": float(cell_lengths[fi, 0]) if fi < len(cell_lengths) else 1.0,
                "b": float(cell_lengths[fi, 1]) if fi < len(cell_lengths) else 1.0,
                "c": float(cell_lengths[fi, 2]) if fi < len(cell_lengths) else 1.0,
                "alpha": float(cell_angles[fi, 0]) if fi < len(cell_angles) else 90.0,
                "beta": float(cell_angles[fi, 1]) if fi < len(cell_angles) else 90.0,
                "gamma": float(cell_angles[fi, 2]) if fi < len(cell_angles) else 90.0,
            }
        )

    return _write_xmolout_from_frames(
        frames,
        out_path,
        simulation_name=str(args.get("simulation") or "MD"),
        precision=int(args.get("precision", 6)),
    )
