"""Projected polarity fractions from basal-plane displacement dipoles."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.dipole import (
    BasalPlaneDipoleRequest,
    BasalPlaneDipoleResult,
    calculate_basal_plane_dipoles,
)
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.neighbors import (
    _frame_labels,
    _source_frame,
    _trajectory_and_charges,
    required_wurtzite_data_type,
)
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import TrajectoryData
from reaxkit.presentation.specs import PresentationSpec

ProjectionPlane = Literal["xy", "xz", "yz"]
CartesianAxis = Literal["x", "y", "z"]


@dataclass
class BasalPlaneProjectedPolarityRequest(BasalPlaneDipoleRequest):
    """Configure spatial averaging of per-center polarity signs."""

    component: CartesianAxis = dc_field(
        default="z", metadata={"label": "Polarity component", "choices": ["x", "y", "z"]}
    )
    projection_plane: ProjectionPlane = dc_field(
        default="xz", metadata={"label": "Projection plane", "choices": ["xy", "xz", "yz"]}
    )
    projection_bins: tuple[int, int] = (40, 40)
    profile_axis: CartesianAxis | None = dc_field(
        default=None, metadata={"label": "Kymograph profile axis", "choices": ["x", "y", "z"]}
    )
    dipole_zero_tolerance: float = 0.0


@dataclass
class BasalPlaneProjectedPolarityResult(BaseResult):
    """Center signs plus spatial and frame-resolved mean-polarity tables."""

    centers: pd.DataFrame
    projected_bins: pd.DataFrame
    kymograph_bins: pd.DataFrame
    request: BasalPlaneProjectedPolarityRequest
    dipole_result: BasalPlaneDipoleResult
    frame_indices: np.ndarray
    iterations: np.ndarray
    projection_edges: tuple[np.ndarray, np.ndarray]
    profile_edges: np.ndarray

    @property
    def table(self) -> pd.DataFrame:
        return self.projected_bins

    @property
    def csv_tables(self) -> dict[str, pd.DataFrame]:
        return {
            "basal_plane_projected_polarity_centers": self.centers,
            "basal_plane_projected_polarity_2d": self.projected_bins,
            "basal_plane_projected_polarity_kymograph": self.kymograph_bins,
        }


def _validate_request(request: BasalPlaneProjectedPolarityRequest) -> None:
    if request.component not in "xyz":
        raise ValueError("component must be 'x', 'y', or 'z'.")
    if request.projection_plane not in {"xy", "xz", "yz"}:
        raise ValueError("projection_plane must be 'xy', 'xz', or 'yz'.")
    if request.profile_axis is not None and request.profile_axis not in request.projection_plane:
        raise ValueError("profile_axis must be one of the two projection-plane axes.")
    if len(request.projection_bins) != 2 or any(int(value) <= 0 for value in request.projection_bins):
        raise ValueError("projection_bins must contain two positive integers.")
    if float(request.dipole_zero_tolerance) < 0.0:
        raise ValueError("dipole_zero_tolerance must be non-negative.")


def _edges(values: np.ndarray, count: int) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.linspace(0.0, 1.0, int(count) + 1)
    lower, upper = float(np.min(finite)), float(np.max(finite))
    if lower == upper:
        lower -= 0.5
        upper += 0.5
    return np.linspace(lower, upper, int(count) + 1)


def _polarity_statistics(
        values: np.ndarray,
) -> tuple[int, int, int, int, float, float, float, float]:
    signs = np.asarray(values, dtype=float)
    signs = signs[np.isfinite(signs)]
    positive = int(np.count_nonzero(signs > 0.0))
    negative = int(np.count_nonzero(signs < 0.0))
    zero = int(np.count_nonzero(signs == 0.0))
    count = positive + negative + zero
    if count == 0:
        return 0, 0, 0, 0, np.nan, np.nan, np.nan, np.nan
    return (
        count,
        positive,
        negative,
        zero,
        float(np.mean(signs)),
        positive / count,
        negative / count,
        zero / count,
    )


def calculate_basal_plane_projected_polarity(
        data,
        request: BasalPlaneProjectedPolarityRequest,
) -> BasalPlaneProjectedPolarityResult:
    """Average fixed-reference-bin center polarities in {-1, 0, +1}."""

    _validate_request(request)
    if request.profile_axis is None:
        request.profile_axis = cast(CartesianAxis, request.projection_plane[1])
    trajectory, _ = _trajectory_and_charges(data)
    dipoles = calculate_basal_plane_dipoles(data, request)
    centers = dipoles.table.copy()
    mu_column = f"mu_{request.component} (e*angstrom)"
    mu = centers[mu_column].to_numpy(float)
    valid = (
            centers["has_basal_plane_dipole"].to_numpy(bool)
            & np.isfinite(mu)
    )
    polarity = np.sign(mu)
    polarity[np.abs(mu) <= float(request.dipole_zero_tolerance)] = 0.0
    centers["polarity_component"] = request.component
    centers["has_defined_polarity"] = valid
    centers["polarity"] = np.where(valid, polarity, np.nan)

    u_axis, v_axis = request.projection_plane
    reference_frame = int(request.reference_frame)
    reference_positions = np.asarray(trajectory.positions[reference_frame], dtype=float)
    reference_labels = _frame_labels(trajectory, reference_frame).astype(str)
    center_names = {str(value).casefold() for value in request.centers}
    center_mask = np.asarray(
        [label.casefold() in center_names for label in reference_labels], dtype=bool
    )
    reference_atom_ids = np.asarray(trajectory.atom_ids, dtype=int)[center_mask]
    reference_coordinates = reference_positions[center_mask]
    u_reference = reference_coordinates[:, "xyz".index(u_axis)]
    v_reference = reference_coordinates[:, "xyz".index(v_axis)]
    u_edges = _edges(u_reference, int(request.projection_bins[0]))
    v_edges = _edges(v_reference, int(request.projection_bins[1]))

    def _bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(edges, values, side="right") - 1
        indices[values == edges[-1]] = len(edges) - 2
        invalid = ~np.isfinite(values) | (values < edges[0]) | (values > edges[-1])
        indices[invalid] = -1
        return indices

    assignments = pd.DataFrame({
        "site_atom_id": reference_atom_ids,
        "reference_u (angstrom)": u_reference,
        "reference_v (angstrom)": v_reference,
        "u_bin": _bin_indices(u_reference, u_edges),
        "v_bin": _bin_indices(v_reference, v_edges),
    })
    centers = centers.merge(assignments, on="site_atom_id", how="left", validate="many_to_one")
    centers["bin_reference_frame"] = _source_frame(trajectory, reference_frame)
    profile_bin_column = "u_bin" if request.profile_axis == u_axis else "v_bin"
    profile_edges = u_edges if request.profile_axis == u_axis else v_edges

    projected_rows: list[dict[str, object]] = []
    kymograph_rows: list[dict[str, object]] = []
    for (frame, iteration), group in centers.groupby(["frame_index", "iter"], sort=True):
        polarity = group["polarity"].to_numpy(float)
        for v_bin in range(len(v_edges) - 1):
            v_member = group["v_bin"].to_numpy(float) == v_bin
            for u_bin in range(len(u_edges) - 1):
                u_member = group["u_bin"].to_numpy(float) == u_bin
                (
                    count, positive, negative, zero, mean,
                    positive_fraction, negative_fraction, zero_fraction,
                ) = _polarity_statistics(polarity[u_member & v_member])
                projected_rows.append({
                    "frame_index": int(frame), "iter": int(iteration),
                    "plane": request.projection_plane,
                    "component": request.component,
                    "u_axis": u_axis, "v_axis": v_axis,
                    "u_bin": u_bin, "v_bin": v_bin,
                    "u_min (angstrom)": float(u_edges[u_bin]),
                    "u_max (angstrom)": float(u_edges[u_bin + 1]),
                    "u_center (angstrom)": float((u_edges[u_bin] + u_edges[u_bin + 1]) / 2.0),
                    "v_min (angstrom)": float(v_edges[v_bin]),
                    "v_max (angstrom)": float(v_edges[v_bin + 1]),
                    "v_center (angstrom)": float((v_edges[v_bin] + v_edges[v_bin + 1]) / 2.0),
                    "defined_center_count": count,
                    "positive_count": positive, "negative_count": negative, "zero_count": zero,
                    "positive_fraction": positive_fraction,
                    "negative_fraction": negative_fraction,
                    "zero_fraction": zero_fraction,
                    "mean_polarity": mean,
                    "bin_reference_frame": _source_frame(trajectory, reference_frame),
                })

        for profile_bin in range(len(profile_edges) - 1):
            member = group[profile_bin_column].to_numpy(float) == profile_bin
            (
                count, positive, negative, zero, mean,
                positive_fraction, negative_fraction, zero_fraction,
            ) = _polarity_statistics(polarity[member])
            kymograph_rows.append({
                "frame_index": int(frame), "iter": int(iteration),
                "profile_axis": request.profile_axis,
                "component": request.component,
                "profile_bin": profile_bin,
                "coordinate_min (angstrom)": float(profile_edges[profile_bin]),
                "coordinate_max (angstrom)": float(profile_edges[profile_bin + 1]),
                "coordinate_center (angstrom)": float(
                    (profile_edges[profile_bin] + profile_edges[profile_bin + 1]) / 2.0
                ),
                "defined_center_count": count,
                "positive_count": positive, "negative_count": negative, "zero_count": zero,
                "positive_fraction": positive_fraction,
                "negative_fraction": negative_fraction,
                "zero_fraction": zero_fraction,
                "mean_polarity": mean,
                "bin_reference_frame": _source_frame(trajectory, reference_frame),
            })

    return BasalPlaneProjectedPolarityResult(
        centers=centers,
        projected_bins=pd.DataFrame(projected_rows),
        kymograph_bins=pd.DataFrame(kymograph_rows),
        request=request,
        dipole_result=dipoles,
        frame_indices=dipoles.frame_indices,
        iterations=dipoles.iterations,
        projection_edges=(u_edges, v_edges),
        profile_edges=profile_edges,
    )


@register_task(
    "get-basal-plane-displacement-projected-polarity",
    label="Basal-plane Projected Polarity",
)
class BasalPlaneProjectedPolarityTask(AnalysisTask):
    """Bin local dipole signs into TEM-like projections and kymographs."""

    required_data = TrajectoryData
    supports_selective_streaming = False
    VERSION = "2"

    def required_data_for(
            self, request: BasalPlaneProjectedPolarityRequest, args: dict | None = None
    ):
        return required_wurtzite_data_type(request, args)

    @staticmethod
    def required_data_fields_for(
            request: BasalPlaneProjectedPolarityRequest, _args: dict
    ) -> tuple[str, ...]:
        return ("trajectory", "charges") if request.charge_source != "formal" else ("trajectory",)

    @staticmethod
    def recommended_presentations(
            _result: BasalPlaneProjectedPolarityResult, _payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        return [
            PresentationSpec(
                renderer="table", label="Projected mean polarity", view_type="table"
            )
        ]

    def run(self, data, request: BasalPlaneProjectedPolarityRequest, reporter=None):
        _ = reporter
        return calculate_basal_plane_projected_polarity(data, request)


__all__ = [
    "BasalPlaneProjectedPolarityRequest",
    "BasalPlaneProjectedPolarityResult",
    "BasalPlaneProjectedPolarityTask",
    "calculate_basal_plane_projected_polarity",
]
