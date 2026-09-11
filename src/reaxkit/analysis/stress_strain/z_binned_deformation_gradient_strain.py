"""Compute reference-to-current deformation-gradient strain in z bins.

Selected frame-zero atoms define material bins. For each reported frame, this
analyzer fits a translation-free affine deformation gradient and derives the
Green-Lagrange strain tensor, engineering shear, principal strains, volume
change, and fit diagnostics. Input loading and rendering are out of scope.

**Usage context**

- Finite deformation: Measure rotation-invariant local strain through a slab.
- Fit diagnostics: Retain rank, conditioning, and residual error per z bin.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis.stress_strain import common
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import TrajectoryData
from reaxkit.presentation.specs import PresentationSpec

F_COLUMNS = tuple(f"F_{row}{column}" for row in "xyz" for column in "xyz")
NORMAL_STRAIN_COLUMNS = ("strain_xx", "strain_yy", "strain_zz")
TENSOR_SHEAR_COLUMNS = ("strain_xy", "strain_xz", "strain_yz")
ENGINEERING_SHEAR_COLUMNS = ("gamma_xy", "gamma_xz", "gamma_yz")
PRINCIPAL_STRAIN_COLUMNS = ("principal_E_1", "principal_E_2", "principal_E_3")
STRAIN_COLUMNS = (*NORMAL_STRAIN_COLUMNS, *TENSOR_SHEAR_COLUMNS, *ENGINEERING_SHEAR_COLUMNS)
RESULT_COLUMNS = [
    "frame", "iter", "bin_number", "bin_mean_z", "number_of_atoms_in_this_bin",
    "fit_rank", "reference_condition_number", "fit_rmse", *F_COLUMNS,
    *NORMAL_STRAIN_COLUMNS, *TENSOR_SHEAR_COLUMNS, *ENGINEERING_SHEAR_COLUMNS,
    *PRINCIPAL_STRAIN_COLUMNS, "J", "volumetric_change",
]


@dataclass
class ZBinnedDeformationGradientStrainRequest(BaseRequest):
    """Request configuration for z-binned affine strain fitting.

    Fields
    -----
    z_bins : int
        Number of equal-width z bins. Default is `20`.
    atom_types : Optional[Sequence[str]]
        Elements to include; `None` selects all atoms. Default is `(\"Al\", \"N\")`.
    bin_range : str
        `\"reference\"` keeps material bins fixed; `\"current\"` reassigns
        atoms by current z. Default is `\"reference\"`.
    minimum_atoms : int
        Minimum atoms needed for a three-dimensional affine fit; at least four.
    max_condition_number : float
        Largest accepted reference geometry condition number.
    selected_frames : Optional[Sequence[int]]
        Zero-based frames to report; `None` reports all frames.
    every : int
        Stride applied to the frame selection. Default is `1`.
    unwrap : bool
        Cumulatively unwrap periodic coordinates. Default is `True`.
    periodic : str
        Periodic axes such as `\"xy\"`, `\"xyz\"`, or `\"none\"`.

    Examples
    -----
    `ZBinnedDeformationGradientStrainRequest(z_bins=10, periodic=\"xy\")`
    configures ten fixed material bins in an xy-periodic slab.
    """

    z_bins: int = dc_field(default=20, metadata={"label": "Z bins", "help": "Number of equal-width z bins.", "min": 1})
    atom_types: Optional[Sequence[str]] = dc_field(default=("Al", "N"), metadata={"label": "Atom types", "help": "Elements to include; empty selects all atoms."})
    bin_range: str = dc_field(default="reference", metadata={"label": "Bin range", "help": "Reference-material or current-position bins.", "choices": ["reference", "current"]})
    minimum_atoms: int = dc_field(default=4, metadata={"label": "Minimum atoms", "help": "Minimum atoms for an affine fit.", "min": 4})
    max_condition_number: float = dc_field(default=1.0e12, metadata={"label": "Maximum condition number", "help": "Reject more ill-conditioned reference fits.", "min": 1.0})
    selected_frames: Optional[Sequence[int]] = dc_field(default=None, metadata={"label": "Frames", "help": "Zero-based frames to report."})
    every: int = dc_field(default=1, metadata={"label": "Frame stride", "help": "Stride over selected frames.", "min": 1})
    unwrap: bool = dc_field(default=True, metadata={"label": "Unwrap PBC", "help": "Cumulatively unwrap periodic coordinates."})
    periodic: str = dc_field(default="xyz", metadata={"label": "Periodic axes", "help": "Periodic axes, such as xy or xyz.", "choices": ["none", "x", "y", "z", "xy", "xz", "yz", "xyz"]})


@dataclass
class ZBinnedDeformationGradientStrainResult(BaseResult):
    """Result of z-binned deformation-gradient strain analysis.

    Fields
    -----
    table : pd.DataFrame
        Per-frame/per-bin deformation gradients, strain measures, and fit
        diagnostics.
    request : ZBinnedDeformationGradientStrainRequest
        Request that produced the table.

    Examples
    -----
    `ZBinnedDeformationGradientStrainResult(table=<DataFrame>, request=request)`
    preserves computed finite-strain values and their configuration.
    """

    table: pd.DataFrame
    request: ZBinnedDeformationGradientStrainRequest


class _PreparedReferenceFit(NamedTuple):
    """Store centered reference geometry and its reusable fit operator."""

    centered_reference: np.ndarray
    fit_operator: np.ndarray | None
    rank: int
    condition_number: float


def _blank_fit_metrics() -> dict[str, float]:
    """Return undefined values for every fitted metric."""
    return {
        **{column: float("nan") for column in F_COLUMNS},
        **{column: float("nan") for column in STRAIN_COLUMNS},
        **{column: float("nan") for column in PRINCIPAL_STRAIN_COLUMNS},
        "J": float("nan"), "volumetric_change": float("nan"), "fit_rmse": float("nan"),
    }


def _prepare_reference_fit(reference_coordinates: np.ndarray, *, minimum_atoms: int, max_condition_number: float) -> _PreparedReferenceFit:
    """Factor reference geometry once and construct its pseudoinverse."""
    reference = np.asarray(reference_coordinates, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference coordinates must have shape (n_atoms, 3).")
    if reference.shape[0] == 0:
        return _PreparedReferenceFit(reference.copy(), None, 0, float("inf"))
    centered = reference - reference.mean(axis=0)
    left, singular_values, right_t = np.linalg.svd(centered, full_matrices=False)
    tolerance = np.finfo(float).eps * max(centered.shape) * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    condition = float(singular_values[0] / singular_values[-1]) if len(singular_values) == 3 and singular_values[-1] > tolerance else float("inf")
    operator = None
    if reference.shape[0] >= minimum_atoms and rank == 3 and condition <= max_condition_number:
        operator = (right_t.T / singular_values) @ left.T
    return _PreparedReferenceFit(centered, operator, rank, condition)


def _metrics_from_deformation_gradient(deformation_gradient: np.ndarray, fit_rmse: float) -> dict[str, float]:
    """Derive reported strain and volume metrics from one fitted matrix."""
    strain = 0.5 * (deformation_gradient.T @ deformation_gradient - np.eye(3))
    principal = np.linalg.eigvalsh(strain)[::-1]
    determinant = float(np.linalg.det(deformation_gradient))
    return {
        **{f"F_{r}{c}": float(deformation_gradient[i, j]) for i, r in enumerate("xyz") for j, c in enumerate("xyz")},
        "strain_xx": float(strain[0, 0]), "strain_yy": float(strain[1, 1]), "strain_zz": float(strain[2, 2]),
        "strain_xy": float(strain[0, 1]), "strain_xz": float(strain[0, 2]), "strain_yz": float(strain[1, 2]),
        "gamma_xy": float(2.0 * strain[0, 1]), "gamma_xz": float(2.0 * strain[0, 2]), "gamma_yz": float(2.0 * strain[1, 2]),
        **{column: float(value) for column, value in zip(PRINCIPAL_STRAIN_COLUMNS, principal)},
        "J": determinant, "volumetric_change": determinant - 1.0, "fit_rmse": fit_rmse,
    }


def _fit_prepared_reference(prepared: _PreparedReferenceFit, current_coordinates: np.ndarray) -> dict[str, float]:
    """Apply a prepared reference fit to current coordinates."""
    current = np.asarray(current_coordinates, dtype=float)
    if current.shape != prepared.centered_reference.shape:
        raise ValueError("current coordinates do not match the reference geometry.")
    if prepared.fit_operator is None:
        return _blank_fit_metrics()
    centered_current = current - current.mean(axis=0)
    fitted_t = prepared.fit_operator @ centered_current
    deformation_gradient = fitted_t.T
    residual = centered_current - prepared.centered_reference @ fitted_t
    fit_rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return _metrics_from_deformation_gradient(deformation_gradient, fit_rmse)


def _fit_deformation_gradient(reference: np.ndarray, current: np.ndarray, *, minimum_atoms: int, max_condition_number: float) -> tuple[dict[str, float], int, float]:
    """Fit a deformation gradient and return metrics plus diagnostics."""
    reference_values, current_values = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    if reference_values.shape != current_values.shape or reference_values.ndim != 2 or reference_values.shape[1] != 3:
        raise ValueError("reference and current coordinates must match with shape (n_atoms, 3).")
    prepared = _prepare_reference_fit(reference_values, minimum_atoms=minimum_atoms, max_condition_number=max_condition_number)
    return _fit_prepared_reference(prepared, current_values), prepared.rank, prepared.condition_number


def calculate_deformation_gradient_strain(data: TrajectoryData, request: ZBinnedDeformationGradientStrainRequest) -> pd.DataFrame:
    """Calculate affine deformation-gradient strain by frame and z bin.

    Works on
    -----
    `TrajectoryData` plus `ZBinnedDeformationGradientStrainRequest` inputs.

    Parameters
    -----
    data : TrajectoryData
        Canonical trajectory and optional simulation cell data.
    request : ZBinnedDeformationGradientStrainRequest
        Selection, fitting, and periodic-boundary controls.

    Returns
    -----
    pd.DataFrame
        Per-frame/per-bin finite-strain rows in `RESULT_COLUMNS` order.

    Examples
    -----
    ```python
    table = calculate_deformation_gradient_strain(data, ZBinnedDeformationGradientStrainRequest(z_bins=10, unwrap=False))
    print(table[["bin_number", "strain_zz"]].head())
    ```
    Sample output reports rotation-invariant normal strain in each bin.
    """
    positions = np.asarray(data.positions, dtype=float)
    if request.minimum_atoms < 4:
        raise ValueError("minimum_atoms must be at least 4.")
    if request.max_condition_number <= 1.0 or not np.isfinite(request.max_condition_number):
        raise ValueError("max_condition_number must be finite and greater than 1.")
    bin_range = str(request.bin_range).lower()
    if bin_range not in {"reference", "current"}:
        raise ValueError("bin_range must be 'reference' or 'current'.")
    frames = common.selected_frames(positions.shape[0], request.selected_frames, request.every)
    selected = common.select_atom_indices(data.elements, request.atom_types)
    eligible = selected[np.all(np.isfinite(positions[0, selected]), axis=1)]
    if eligible.size == 0:
        raise ValueError("No selected atoms have finite coordinates in frame 0.")
    reference = positions[0, eligible]
    reference_edges = common.bin_edges(reference[:, 2], request.z_bins)
    reference_bins = common.assign_bins(reference[:, 2], reference_edges)
    reference_indices = [np.flatnonzero(reference_bins == number) for number in range(1, request.z_bins + 1)]
    prepared = [
        _prepare_reference_fit(reference[index], minimum_atoms=request.minimum_atoms, max_condition_number=request.max_condition_number)
        for index in reference_indices
    ]
    iterations = common.iteration_values(data)
    rows: list[dict[str, int | float]] = []
    for frame, current_all, current_valid in common.iter_selected_coordinates(
        data, eligible, frames, unwrap=request.unwrap, periodic=request.periodic
    ):
        if bin_range == "reference":
            edges, frame_bins = reference_edges, reference_bins
        else:
            finite = current_all[current_valid]
            if finite.size == 0:
                raise ValueError(f"Frame {frame} contains no finite selected atoms.")
            edges = common.bin_edges(finite[:, 2], request.z_bins)
            frame_bins = np.zeros(eligible.size, dtype=int)
            frame_bins[current_valid] = common.assign_bins(finite[:, 2], edges)
        for zero_based_bin, midpoint in enumerate((edges[:-1] + edges[1:]) / 2.0):
            bin_number = zero_based_bin + 1
            if bin_range == "reference":
                all_indices = reference_indices[zero_based_bin]
                valid_indices = all_indices[current_valid[all_indices]]
                if len(valid_indices) == len(all_indices):
                    metrics = _fit_prepared_reference(prepared[zero_based_bin], current_all[all_indices])
                    rank, condition = prepared[zero_based_bin].rank, prepared[zero_based_bin].condition_number
                else:
                    metrics, rank, condition = _fit_deformation_gradient(reference[valid_indices], current_all[valid_indices], minimum_atoms=request.minimum_atoms, max_condition_number=request.max_condition_number)
                atom_count = len(valid_indices)
            else:
                in_bin = (frame_bins == bin_number) & current_valid
                atom_count = int(in_bin.sum())
                metrics, rank, condition = _fit_deformation_gradient(reference[in_bin], current_all[in_bin], minimum_atoms=request.minimum_atoms, max_condition_number=request.max_condition_number)
            rows.append({
                "frame": frame, "iter": int(iterations[frame]), "bin_number": bin_number,
                "bin_mean_z": float(midpoint), "number_of_atoms_in_this_bin": int(atom_count),
                "fit_rank": int(rank), "reference_condition_number": condition, **metrics,
            })
    if not rows:
        raise ValueError("No requested frames were analyzed.")
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


@register_task("get_z_binned_deformation_gradient_strain", label="Z-Binned Deformation-Gradient Strain")
class ZBinnedDeformationGradientStrainTask(AnalysisTask):
    """Run z-binned deformation-gradient strain analysis."""

    required_data = TrajectoryData

    @staticmethod
    def recommended_presentations(_result: ZBinnedDeformationGradientStrainResult, payload: dict[str, Any]) -> list[PresentationSpec]:
        """Return table and normal-strain profile presentation specifications.

        Works on
        -----
        Analyzer task output payloads.

        Parameters
        -----
        _result : ZBinnedDeformationGradientStrainResult
            Typed analyzer result.
        payload : dict[str, Any]
            Serialized result payload.

        Returns
        -----
        list[PresentationSpec]
            Table and strain-versus-z plot specifications.

        Examples
        -----
        `ZBinnedDeformationGradientStrainTask.recommended_presentations(result, payload)`
        returns standard downstream views.
        """
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(renderer="single_plot", label="Normal Strain vs Z", mapping={"x_col": "bin_mean_z", "y_col": "strain_zz", "group_by_col": "frame"}, options={"title": "Z-Binned Deformation-Gradient Strain", "xlabel": "Bin mean z", "ylabel": "strain_zz", "legend": True}, view_type="plot2d"),
        ]

    def run(self, data: TrajectoryData, request: ZBinnedDeformationGradientStrainRequest, reporter=None) -> ZBinnedDeformationGradientStrainResult:
        """Execute z-binned deformation-gradient strain analysis.

        Works on
        -----
        `TrajectoryData` plus `ZBinnedDeformationGradientStrainRequest` inputs.

        Parameters
        -----
        data : TrajectoryData
            Canonical trajectory data.
        request : ZBinnedDeformationGradientStrainRequest
            Analysis configuration.
        reporter : Any, optional
            Progress reporter accepted by the analyzer interface.

        Returns
        -----
        ZBinnedDeformationGradientStrainResult
            Computed table and request provenance.

        Examples
        -----
        `ZBinnedDeformationGradientStrainTask().run(data, request)` returns
        fitted strain and diagnostics for every selected bin.
        """
        _ = reporter
        return ZBinnedDeformationGradientStrainResult(table=calculate_deformation_gradient_strain(data, request), request=request)
