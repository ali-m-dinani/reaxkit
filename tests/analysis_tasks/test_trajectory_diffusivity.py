"""Unit tests for trajectory diffusivity analysis."""

from __future__ import annotations

import numpy as np

from reaxkit.analysis.trajectory.diffusivity import DiffusivityRequest, DiffusivityTask
from reaxkit.analysis.trajectory.msd import MSDRequest, MSDTask
from reaxkit.domain.data_models import TrajectoryData
from reaxkit.domain.data_models import SimulationData


def _trajectory_linear_msd() -> TrajectoryData:
    # For atom 1: MSD(t) = 3*t by construction using r(t) = [sqrt(3*t), 0, 0].
    times = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)
    atom1 = np.asarray([[np.sqrt(3.0 * t), 0.0, 0.0] for t in times], dtype=float)
    atom2 = np.asarray([[0.0, 0.0, 0.0] for _ in times], dtype=float)
    positions = np.stack([np.stack([a1, a2], axis=0) for a1, a2 in zip(atom1, atom2)], axis=0)
    sim = SimulationData(
        atom_ids=[1, 2],
        iterations=np.asarray([0, 1, 2, 3], dtype=int),
        cell_lengths=np.asarray([[100.0, 100.0, 100.0]] * 4, dtype=float),
        cell_angles=np.asarray([[90.0, 90.0, 90.0]] * 4, dtype=float),
    )
    return TrajectoryData(
        positions=positions,
        elements=["H", "He"],
        atom_ids=[1, 2],
        iterations=np.asarray([0, 1, 2, 3], dtype=int),
        simulation=sim,
    )


def test_diffusivity_task_uses_einstein_relation_with_default_d() -> None:
    data = _trajectory_linear_msd()
    result = DiffusivityTask().run(data, DiffusivityRequest(atom_ids=[1]))

    assert result.table["atom_id"].tolist() == [1]
    assert result.table["x_source"].tolist() == ["iter"]
    assert np.isclose(float(result.table["slope_msd_per_x"].iloc[0]), 3.0, atol=1e-6)
    assert np.isclose(float(result.table["diffusivity"].iloc[0]), 0.5, atol=1e-6)


def test_diffusivity_task_respects_custom_d() -> None:
    data = _trajectory_linear_msd()
    result = DiffusivityTask().run(data, DiffusivityRequest(atom_ids=[1], d=1.0))

    assert np.isclose(float(result.table["diffusivity"].iloc[0]), 1.5, atol=1e-6)


def _trajectory_wrapped_boundary_crossing() -> TrajectoryData:
    # Atom 1 crosses x-boundary in a 10 A periodic box:
    # wrapped x: 1 -> 9 corresponds to an unwrapped displacement of +2 A.
    positions = np.asarray(
        [
            [[1.0, 0.0, 0.0]],
            [[9.0, 0.0, 0.0]],
        ],
        dtype=float,
    )
    sim = SimulationData(
        atom_ids=[1],
        iterations=np.asarray([0, 1], dtype=int),
        cell_lengths=np.asarray([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]], dtype=float),
        cell_angles=np.asarray([[90.0, 90.0, 90.0], [90.0, 90.0, 90.0]], dtype=float),
    )
    return TrajectoryData(
        positions=positions,
        elements=["H"],
        atom_ids=[1],
        iterations=np.asarray([0, 1], dtype=int),
        simulation=sim,
    )


def test_msd_task_unwraps_boundary_crossing_by_default() -> None:
    data = _trajectory_wrapped_boundary_crossing()
    result = MSDTask().run(data, MSDRequest(atom_ids=[1], unwrap=True))
    assert np.isclose(float(result.table["msd"].iloc[-1]), 4.0, atol=1e-6)


def test_diffusivity_task_uses_unwrapped_displacement() -> None:
    data = _trajectory_wrapped_boundary_crossing()
    # For unwrapped displacement +2 A at dt=1, MSD slope is 4 and D = slope/(2*d) = 4/6.
    result = DiffusivityTask().run(data, DiffusivityRequest(atom_ids=[1], d=3.0, unwrap=True))
    assert np.isclose(float(result.table["slope_msd_per_x"].iloc[0]), 4.0, atol=1e-6)
    assert np.isclose(float(result.table["diffusivity"].iloc[0]), 4.0 / 6.0, atol=1e-6)
