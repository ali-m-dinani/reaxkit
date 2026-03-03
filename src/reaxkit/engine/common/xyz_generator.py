"""Simple XYZ trajectory writer for engine-agnostic fallback export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reaxkit.domain.data_models import TrajectoryData


def write_xyz_trajectory(
    trajectory: TrajectoryData,
    out_path: str | Path,
    *,
    precision: int = 6,
) -> Path:
    """Write a ``TrajectoryData`` object as a multi-frame XYZ file."""
    positions = np.asarray(trajectory.positions, dtype=float)
    if positions.ndim != 3:
        raise ValueError("TrajectoryData.positions must have shape (n_frames, n_atoms, 3).")

    n_frames, n_atoms, _ = positions.shape
    labels = (
        np.asarray(trajectory.atom_labels, dtype=object)
        if trajectory.atom_labels is not None
        else np.tile(np.asarray(trajectory.elements, dtype=object), (n_frames, 1))
    )
    if labels.shape != (n_frames, n_atoms):
        raise ValueError("TrajectoryData atom labels must have shape (n_frames, n_atoms).")

    iterations = (
        np.asarray(trajectory.iterations, dtype=int)
        if trajectory.iterations is not None
        else np.arange(n_frames, dtype=int)
    )
    fmt = f"{{:.{precision}f}}"
    lines: list[str] = []
    for fi in range(n_frames):
        lines.append(f"{n_atoms}\n")
        iter_value = int(iterations[fi]) if fi < len(iterations) else int(fi)
        lines.append(f"frame={fi} iter={iter_value}\n")
        for atom_label, xyz in zip(labels[fi].tolist(), positions[fi], strict=False):
            lines.append(
                f"{str(atom_label)} {fmt.format(float(xyz[0]))} {fmt.format(float(xyz[1]))} {fmt.format(float(xyz[2]))}\n"
            )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")
    return out


__all__ = ["write_xyz_trajectory"]
