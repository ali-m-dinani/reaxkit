"""Unit tests for trajectory dihedral analysis."""

from __future__ import annotations

import numpy as np

from reaxkit.analysis.trajectory.dihedral import DihedralRequest, DihedralTask, calculate_dihedral_numpy
from reaxkit.domain.data_models import TrajectoryData


def _make_dihedral_points(angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.deg2rad(float(angle_deg))
    p0 = np.array([1.0, 0.0, 0.0], dtype=float)
    p1 = np.array([0.0, 0.0, 0.0], dtype=float)
    p2 = np.array([0.0, 1.0, 0.0], dtype=float)
    p3 = np.array([np.cos(theta), 1.0, np.sin(theta)], dtype=float)
    return p0, p1, p2, p3


def test_calculate_dihedral_numpy_returns_expected_magnitude() -> None:
    p0, p1, p2, p3 = _make_dihedral_points(60.0)
    angle = float(calculate_dihedral_numpy(p0, p1, p2, p3, degrees=True))
    assert np.isclose(abs(angle), 60.0, atol=1e-6)


def test_dihedral_task_runs_on_trajectory_data() -> None:
    p0, p1, p2, p3 = _make_dihedral_points(60.0)
    q0, q1, q2, q3 = _make_dihedral_points(-45.0)
    positions = np.asarray(
        [
            [p0, p1, p2, p3],
            [q0, q1, q2, q3],
        ],
        dtype=float,
    )

    data = TrajectoryData(
        positions=positions,
        elements=["C", "C", "C", "C"],
        atom_ids=[1, 2, 3, 4],
        iterations=np.array([10, 20], dtype=int),
    )
    request = DihedralRequest(atom_ids=[1, 2, 3, 4], units="deg", backend="numpy")

    result = DihedralTask().run(data, request)
    assert list(result.table.columns) == [
        "frame_index",
        "iter",
        "atom1_id",
        "atom2_id",
        "atom3_id",
        "atom4_id",
        "dihedral",
        "units",
        "backend",
    ]
    assert result.table["frame_index"].tolist() == [0, 1]
    assert result.table["iter"].tolist() == [10, 20]
    assert np.isclose(abs(float(result.table["dihedral"].iloc[0])), 60.0, atol=1e-6)
    assert np.isclose(abs(float(result.table["dihedral"].iloc[1])), 45.0, atol=1e-6)
