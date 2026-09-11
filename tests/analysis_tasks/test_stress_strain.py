from __future__ import annotations

import numpy as np

from reaxkit.analysis.stress_strain.z_binned_deformation_gradient_strain import (
    ZBinnedDeformationGradientStrainRequest,
    ZBinnedDeformationGradientStrainTask,
)
from reaxkit.analysis.stress_strain.z_binned_strain_using_top_bottom_atoms import (
    ZBinnedTopBottomStrainRequest,
    ZBinnedTopBottomStrainTask,
)
from reaxkit.domain.data_models import SimulationData, TrajectoryData


def _span_trajectory() -> TrajectoryData:
    reference = []
    for atom_index in range(20):
        x = float(atom_index % 10)
        y = 10.0 + 2.0 * x
        z = 0.25 + 0.45 * (atom_index % 10) + (5.0 if atom_index >= 10 else 0.0)
        reference.append([x, y, z])
    frame_zero = np.asarray(reference, dtype=float)
    frame_one = frame_zero * np.asarray([1.10, 0.90, 1.20])
    return TrajectoryData(
        positions=np.stack([frame_zero, frame_one]),
        elements=["Al"] * len(frame_zero),
        atom_ids=list(range(1, len(frame_zero) + 1)),
        iterations=np.asarray([0, 100]),
    )


def _reference_cloud() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.2], [0.0, 1.0, 0.4],
            [1.0, 1.0, 0.7], [0.2, 0.3, 1.0], [1.3, 0.4, 1.2],
            [0.4, 1.4, 1.5], [1.5, 1.2, 1.8],
        ],
        dtype=float,
    )


def test_top_bottom_task_recovers_axis_scaling() -> None:
    request = ZBinnedTopBottomStrainRequest(
        z_bins=2,
        atom_types=("Al",),
        n_extreme_atoms=5,
        unwrap=False,
    )
    result = ZBinnedTopBottomStrainTask().run(_span_trajectory(), request)
    frame_one = result.table[result.table["frame"] == 1]

    np.testing.assert_allclose(frame_one["strain_xx"], 0.10, atol=1.0e-12)
    np.testing.assert_allclose(frame_one["strain_yy"], -0.10, atol=1.0e-12)
    np.testing.assert_allclose(frame_one["strain_zz"], 0.20, atol=1.0e-12)
    assert frame_one["number_of_atoms_in_this_bin"].tolist() == [10, 10]


def test_deformation_gradient_task_recovers_affine_map() -> None:
    reference = _reference_cloud()
    deformation_gradient = np.asarray(
        [[1.10, 0.08, 0.00], [0.00, 0.90, 0.03], [0.02, 0.00, 1.20]],
        dtype=float,
    )
    current = reference @ deformation_gradient.T + np.asarray([7.0, -4.0, 2.5])
    data = TrajectoryData(
        positions=np.stack([reference, current]),
        elements=["Al"] * len(reference),
        atom_ids=list(range(1, len(reference) + 1)),
    )
    request = ZBinnedDeformationGradientStrainRequest(
        z_bins=1,
        atom_types=("Al",),
        unwrap=False,
    )
    row = ZBinnedDeformationGradientStrainTask().run(data, request).table.query("frame == 1").iloc[0]
    recovered = np.asarray([[row[f"F_{r}{c}"] for c in "xyz"] for r in "xyz"])

    np.testing.assert_allclose(recovered, deformation_gradient, atol=1.0e-12)
    assert abs(row["J"] - np.linalg.det(deformation_gradient)) < 1.0e-12
    assert abs(row["fit_rmse"]) < 1.0e-12


def test_deformation_gradient_is_rotation_invariant() -> None:
    reference = _reference_cloud()
    angle = np.radians(31.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    data = TrajectoryData(
        positions=np.stack([reference, reference @ rotation.T + [3.0, 2.0, -1.0]]),
        elements=["Al"] * len(reference),
        atom_ids=list(range(1, len(reference) + 1)),
    )
    request = ZBinnedDeformationGradientStrainRequest(z_bins=1, atom_types=("Al",), unwrap=False)
    row = ZBinnedDeformationGradientStrainTask().run(data, request).table.query("frame == 1").iloc[0]

    np.testing.assert_allclose(
        row[["strain_xx", "strain_yy", "strain_zz", "gamma_xy", "gamma_xz", "gamma_yz"]].to_numpy(dtype=float),
        0.0,
        atol=1.0e-12,
    )


def test_top_bottom_unwrap_removes_periodic_boundary_jump() -> None:
    reference = np.asarray(
        [[8.0 + 0.2 * atom, 1.0 + atom, 1.0 + 0.1 * atom] for atom in range(10)],
        dtype=float,
    )
    wrapped = reference.copy()
    wrapped[:, 0] = np.mod(wrapped[:, 0] + 1.0, 10.0)
    atom_ids = list(range(1, 11))
    simulation = SimulationData(
        atom_ids=atom_ids,
        iterations=np.asarray([0, 1]),
        cell_lengths=np.asarray([[10.0, 20.0, 20.0], [10.0, 20.0, 20.0]]),
        cell_angles=np.asarray([[90.0, 90.0, 90.0], [90.0, 90.0, 90.0]]),
    )
    data = TrajectoryData(
        positions=np.stack([reference, wrapped]),
        elements=["Al"] * 10,
        atom_ids=atom_ids,
        simulation=simulation,
    )
    request = ZBinnedTopBottomStrainRequest(
        z_bins=1,
        atom_types=("Al",),
        n_extreme_atoms=5,
        unwrap=True,
        periodic="x",
    )
    row = ZBinnedTopBottomStrainTask().run(data, request).table.query("frame == 1").iloc[0]

    assert abs(row["span_change_x"]) < 1.0e-12
    assert abs(row["strain_xx"]) < 1.0e-12
