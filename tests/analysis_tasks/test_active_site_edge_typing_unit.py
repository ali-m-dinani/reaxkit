"""Unit checks for phase-2 active-site edge typing."""

from __future__ import annotations

import numpy as np

from reaxkit.analysis.active_sites import ActiveSiteStructuralRequest, ActiveSiteStructuralTask
from reaxkit.domain.data_models import ConnectivityData, ConnectivityTrajectoryData, TrajectoryData


def _run_structural(positions: np.ndarray, bo_frame: np.ndarray):
    n_atoms = positions.shape[1]
    traj = TrajectoryData(
        positions=positions,
        elements=["C"] * n_atoms,
        atom_ids=list(range(1, n_atoms + 1)),
        iterations=np.array([0], dtype=int),
    )
    conn = ConnectivityData(
        bond_orders=np.asarray([bo_frame], dtype=float),
        atom_ids=list(range(1, n_atoms + 1)),
        elements=["C"] * n_atoms,
        iterations=np.array([0], dtype=int),
    )
    data = ConnectivityTrajectoryData(connectivity=conn, trajectory=traj)
    task = ActiveSiteStructuralTask()
    req = ActiveSiteStructuralRequest(frame=0, bo_threshold=0.5, include_noncarbon=True)
    return task.run(data, req)


def test_zz_typing_detected_from_singh_rule():
    # Atom 0 has degree 2; its two neighbors (1,2) each have degree 3 -> zigzag.
    n = 7
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.6, 0.8, 0.0],
            [1.6, -0.8, 0.0],
            [-1.6, 0.8, 0.0],
            [-1.6, -0.8, 0.0],
        ],
        dtype=float,
    ).reshape(1, n, 3)
    bo = np.zeros((n, n), dtype=float)
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    for i, j in edges:
        bo[i, j] = bo[j, i] = 1.0

    result = _run_structural(pos, bo)
    row0 = result.table[result.table["atom_id"] == 1].iloc[0]
    assert row0["edge_label"] == "edge_zigzag"
    assert row0["label"] == "edge_zigzag"
    assert int(row0["seg_id"]) >= 0


def test_ac_typing_detected_from_singh_rule():
    # Atom 0 has degree 2; neighbor degrees are {2,3} -> armchair.
    n = 6
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.8, 0.0, 0.0],
            [-1.8, 0.8, 0.0],
            [-1.8, -0.8, 0.0],
        ],
        dtype=float,
    ).reshape(1, n, 3)
    bo = np.zeros((n, n), dtype=float)
    edges = [(0, 1), (0, 2), (1, 3), (2, 4), (2, 5)]
    for i, j in edges:
        bo[i, j] = bo[j, i] = 1.0

    result = _run_structural(pos, bo)
    row0 = result.table[result.table["atom_id"] == 1].iloc[0]
    assert row0["edge_label"] == "edge_armchair"
    assert row0["label"] == "edge_armchair"
    assert int(row0["seg_id"]) >= 0
