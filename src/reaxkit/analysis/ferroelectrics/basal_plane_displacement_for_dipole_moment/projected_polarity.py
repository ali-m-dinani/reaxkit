"""Fast projected polarity fractions from basal-plane displacement dipoles."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field as dc_field
from typing import Any, ClassVar, Iterable, Literal, cast

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.dipole import (
    BasalPlaneDipoleRequest,
    BasalPlaneDipoleResult,
    ELECTRON_CHARGE_SIGN,
)
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.neighbors import (
    _formal_charge_map,
    _frame_cell,
    _frame_labels,
    _source_frame,
    _trajectory_and_charges,
    _validate_request as _validate_neighbor_request,
    required_wurtzite_data_type,
    unit_vector,
)
from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.neighbors import (
    all_neighbors_within_cutoff,
)
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.core.runtime.execution_contracts import (
    ExecutionShape,
    TaskCapabilities,
    resolve_execution_policy,
)
from reaxkit.core.runtime.reference_frames import reference_frames
from reaxkit.core.runtime.frame_pipeline import BoundedFramePipeline
from reaxkit.core.runtime.reducers import TableAccumulator
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import TrajectoryData
from reaxkit.presentation.specs import PresentationSpec

ProjectionPlane = Literal["xy", "xz", "yz"]
CartesianAxis = Literal["x", "y", "z"]

_PIPELINE_CAPABILITIES = TaskCapabilities(
    shape=ExecutionShape.REFERENCE_FRAME_MAP,
    thread_safe=True,
    needs_reference=True,
    reference_fields=("reference_frame",),
    supports_selective_frames=True,
    estimated_frame_bytes=16 * 1024 * 1024,
)


class _PipelineContract:
    execution_capabilities = _PIPELINE_CAPABILITIES


@dataclass
class BasalPlaneProjectedPolarityRequest(BasalPlaneDipoleRequest):
    """Configure spatial averaging of per-center polarity signs."""

    stream_reference_first: ClassVar[bool] = True
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
    include_centers: bool = True
    workers: int = 0
    chunk_size: int = 0


@dataclass
class BasalPlaneProjectedPolarityResult(BaseResult):
    """Compact projected and frame-resolved mean-polarity tables."""

    artifact_tiers = {
        "basal_plane_projected_polarity_2d": "core",
        "basal_plane_projected_polarity_kymograph": "core",
        "basal_plane_projected_polarity_centers": "detail",
    }

    centers: pd.DataFrame
    projected_bins: pd.DataFrame
    kymograph_bins: pd.DataFrame
    request: BasalPlaneProjectedPolarityRequest
    dipole_result: BasalPlaneDipoleResult | None
    frame_indices: np.ndarray
    iterations: np.ndarray
    projection_edges: tuple[np.ndarray, np.ndarray]
    profile_edges: np.ndarray

    @property
    def table(self) -> pd.DataFrame:
        return self.projected_bins

    @property
    def csv_tables(self) -> dict[str, pd.DataFrame]:
        tables = {
            "basal_plane_projected_polarity_2d": self.projected_bins,
            "basal_plane_projected_polarity_kymograph": self.kymograph_bins,
        }
        if not self.centers.empty:
            tables["basal_plane_projected_polarity_centers"] = self.centers
        return tables


@dataclass(frozen=True)
class _ProjectionContext:
    reference_frame: int
    center_index_by_id: dict[int, int]
    reference_coordinates: np.ndarray
    u_edges: np.ndarray
    v_edges: np.ndarray
    u_bins: np.ndarray
    v_bins: np.ndarray
    profile_bins: np.ndarray
    profile_edges: np.ndarray
    c_hat: np.ndarray
    formal_charges: dict[str, float]


@dataclass(frozen=True)
class _FramePayload:
    frame_index: int
    iteration: int
    xyz: np.ndarray
    labels: np.ndarray
    atom_ids: np.ndarray
    cell: np.ndarray | None
    charges: np.ndarray | None


def _validate_request(request: BasalPlaneProjectedPolarityRequest) -> None:
    _validate_neighbor_request(request)
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
    if int(request.workers) < 0:
        raise ValueError("workers must be zero (automatic) or at least 1.")
    if int(request.chunk_size) < 0:
        raise ValueError("chunk_size must be zero (automatic) or at least 1.")


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


def _bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(edges, values, side="right") - 1
    indices[values == edges[-1]] = len(edges) - 2
    invalid = ~np.isfinite(values) | (values < edges[0]) | (values > edges[-1])
    indices[invalid] = -1
    return indices


def _iteration(trajectory: TrajectoryData, frame: int) -> int:
    values = trajectory.iterations
    if values is None and trajectory.simulation is not None:
        values = trajectory.simulation.iterations
    return int(np.asarray(values).reshape(-1)[frame]) if values is not None else int(frame)


def _selected_frames(trajectory: TrajectoryData, request) -> list[int]:
    count = int(np.asarray(trajectory.positions).shape[0])
    selected = list(range(count)) if request.frames is None else [int(value) for value in request.frames]
    invalid = [value for value in selected if value < 0 or value >= count]
    if invalid:
        raise ValueError(f"Frame(s) not found in trajectory: {invalid}.")
    return selected[:: max(1, int(request.every))]


def _payload(data, frame: int, output_frame: int, request) -> _FramePayload:
    trajectory, raw_charges = _trajectory_and_charges(data)
    charges = None
    if request.charge_source != "formal" and raw_charges is not None:
        charges = np.asarray(raw_charges[frame], dtype=float)
    return _FramePayload(
        frame_index=int(output_frame),
        iteration=_iteration(trajectory, frame),
        xyz=np.asarray(trajectory.positions[frame], dtype=float),
        labels=_frame_labels(trajectory, frame),
        atom_ids=np.asarray(trajectory.atom_ids, dtype=int),
        cell=_frame_cell(trajectory, request, frame),
        charges=charges,
    )


def _build_context(payload: _FramePayload, request) -> _ProjectionContext:
    folded = np.asarray([str(value).casefold() for value in payload.labels], dtype=object)
    center_names = {str(value).casefold() for value in request.centers}
    mask = np.isfinite(payload.xyz).all(axis=1) & np.isin(folded, list(center_names))
    center_ids = payload.atom_ids[mask]
    reference_coordinates = payload.xyz[mask]
    if center_ids.size == 0:
        raise ValueError(f"Frame {payload.frame_index} contains no selected centers.")
    u_axis, v_axis = request.projection_plane
    u_values = reference_coordinates[:, "xyz".index(u_axis)]
    v_values = reference_coordinates[:, "xyz".index(v_axis)]
    u_edges = _edges(u_values, int(request.projection_bins[0]))
    v_edges = _edges(v_values, int(request.projection_bins[1]))
    u_bins = _bin_indices(u_values, u_edges)
    v_bins = _bin_indices(v_values, v_edges)
    profile_bins, profile_edges = (
        (u_bins, u_edges) if request.profile_axis == u_axis else (v_bins, v_edges)
    )
    return _ProjectionContext(
        reference_frame=payload.frame_index,
        center_index_by_id={int(atom_id): index for index, atom_id in enumerate(center_ids)},
        reference_coordinates=reference_coordinates,
        u_edges=u_edges,
        v_edges=v_edges,
        u_bins=u_bins,
        v_bins=v_bins,
        profile_bins=profile_bins,
        profile_edges=profile_edges,
        c_hat=unit_vector(request.c_axis, "c_axis"),
        formal_charges=_formal_charge_map(request),
    )


def _aggregate(signs: np.ndarray, bins: np.ndarray, count: int):
    valid = np.isfinite(signs) & (bins >= 0) & (bins < count)
    selected_bins = bins[valid]
    selected = signs[valid]
    total = np.bincount(selected_bins, minlength=count)
    positive = np.bincount(selected_bins, weights=selected > 0.0, minlength=count).astype(int)
    negative = np.bincount(selected_bins, weights=selected < 0.0, minlength=count).astype(int)
    zero = np.bincount(selected_bins, weights=selected == 0.0, minlength=count).astype(int)
    sums = np.bincount(selected_bins, weights=selected, minlength=count)
    mean = np.divide(sums, total, out=np.full(count, np.nan), where=total > 0)
    fraction_positive = np.divide(positive, total, out=np.full(count, np.nan), where=total > 0)
    fraction_negative = np.divide(negative, total, out=np.full(count, np.nan), where=total > 0)
    fraction_zero = np.divide(zero, total, out=np.full(count, np.nan), where=total > 0)
    return total, positive, negative, zero, mean, fraction_positive, fraction_negative, fraction_zero


def _frame_rows(payload: _FramePayload, request, context: _ProjectionContext):
    xyz = payload.xyz
    labels = payload.labels.astype(str)
    folded = np.asarray([value.casefold() for value in labels], dtype=object)
    finite = np.isfinite(xyz).all(axis=1)
    center_names = {str(value).casefold() for value in request.centers}
    neighbor_names = {str(value).casefold() for value in request.neighbors}
    center_indices = np.flatnonzero(finite & np.isin(folded, list(center_names)))
    neighbor_indices = np.flatnonzero(finite & np.isin(folded, list(neighbor_names)))
    if center_indices.size == 0:
        raise ValueError(f"Frame {payload.frame_index} contains no selected centers.")
    if neighbor_indices.size < 3:
        raise ValueError(
            f"Frame {payload.frame_index} contains only {neighbor_indices.size} selected neighbor atoms; at least three are required."
        )
    periodic = tuple(bool(value) for value in request.periodic)
    frame_periodic = periodic if payload.cell is not None else (False, False, False)
    if any(periodic) and payload.cell is None:
        warnings.warn(
            "Periodic directions were requested but no cell was supplied; using non-periodic distances.",
            RuntimeWarning,
            stacklevel=2,
        )
    candidates = all_neighbors_within_cutoff(
        xyz[center_indices],
        xyz[neighbor_indices],
        cell=payload.cell,
        periodic=frame_periodic,
        cutoff=request.neighbor_cutoff,
    )
    component_index = "xyz".index(request.component)
    center_ids = payload.atom_ids[center_indices]
    center_labels = labels[center_indices]
    center_charges = (
        payload.charges[center_indices]
        if payload.charges is not None
        else np.asarray(
            [context.formal_charges.get(value.casefold(), np.nan) for value in center_labels],
            dtype=float,
        )
    )
    displacement = np.full(len(center_indices), np.nan)
    has_dipole = np.zeros(len(center_indices), dtype=bool)
    for center_row, center_candidates in enumerate(candidates):
        projections = np.asarray([candidate[2] @ context.c_hat for candidate in center_candidates])
        basal = sorted(
            range(len(center_candidates)),
            key=lambda position: (abs(float(projections[position])), center_candidates[position][1]),
        )[:3]
        if len(basal) != 3:
            continue
        mean_basal_vector_component = float(
            np.mean([center_candidates[position][2][component_index] for position in basal])
        )
        displacement[center_row] = -mean_basal_vector_component
        has_dipole[center_row] = np.isfinite(center_charges[center_row])
    dipole = ELECTRON_CHARGE_SIGN * center_charges * displacement
    valid = has_dipole & np.isfinite(dipole)
    signs = np.sign(dipole)
    signs[np.abs(dipole) <= float(request.dipole_zero_tolerance)] = 0.0
    signs[~valid] = np.nan

    reference_indices = np.fromiter(
        (context.center_index_by_id.get(int(atom_id), -1) for atom_id in center_ids),
        dtype=int,
        count=len(center_ids),
    )
    valid_reference = reference_indices >= 0
    u_bins = np.full(len(center_ids), -1, dtype=int)
    v_bins = np.full(len(center_ids), -1, dtype=int)
    profile_bins = np.full(len(center_ids), -1, dtype=int)
    u_bins[valid_reference] = context.u_bins[reference_indices[valid_reference]]
    v_bins[valid_reference] = context.v_bins[reference_indices[valid_reference]]
    profile_bins[valid_reference] = context.profile_bins[reference_indices[valid_reference]]

    center_rows = []
    if request.include_centers:
        for row_index, atom_index in enumerate(center_indices):
            reference_index = reference_indices[row_index]
            reference = (
                context.reference_coordinates[reference_index]
                if reference_index >= 0
                else np.full(3, np.nan)
            )
            center_rows.append({
                "frame_index": payload.frame_index,
                "iter": payload.iteration,
                "site_atom_index": int(atom_index),
                "site_atom_id": int(center_ids[row_index]),
                "site_element": str(center_labels[row_index]),
                "site_x (angstrom)": float(xyz[atom_index, 0]),
                "site_y (angstrom)": float(xyz[atom_index, 1]),
                "site_z (angstrom)": float(xyz[atom_index, 2]),
                "center_charge (e)": float(center_charges[row_index]),
                f"center_displacement_{request.component} (angstrom)": float(displacement[row_index]),
                f"mu_{request.component} (e*angstrom)": float(dipole[row_index]),
                "has_basal_plane_dipole": bool(valid[row_index]),
                "polarity_component": request.component,
                "has_defined_polarity": bool(np.isfinite(signs[row_index])),
                "polarity": float(signs[row_index]),
                "reference_u (angstrom)": float(reference["xyz".index(request.projection_plane[0])]),
                "reference_v (angstrom)": float(reference["xyz".index(request.projection_plane[1])]),
                "u_bin": int(u_bins[row_index]),
                "v_bin": int(v_bins[row_index]),
                "bin_reference_frame": context.reference_frame,
            })

    nu, nv = (int(value) for value in request.projection_bins)
    flat_bins = np.where((u_bins >= 0) & (v_bins >= 0), v_bins * nu + u_bins, -1)
    aggregates = _aggregate(signs, flat_bins, nu * nv)
    projected_rows = []
    u_axis, v_axis = request.projection_plane
    for v_bin in range(nv):
        for u_bin in range(nu):
            index = v_bin * nu + u_bin
            projected_rows.append({
                "frame_index": payload.frame_index,
                "iter": payload.iteration,
                "plane": request.projection_plane,
                "component": request.component,
                "u_axis": u_axis,
                "v_axis": v_axis,
                "u_bin": u_bin,
                "v_bin": v_bin,
                "u_min (angstrom)": float(context.u_edges[u_bin]),
                "u_max (angstrom)": float(context.u_edges[u_bin + 1]),
                "u_center (angstrom)": float((context.u_edges[u_bin] + context.u_edges[u_bin + 1]) / 2.0),
                "v_min (angstrom)": float(context.v_edges[v_bin]),
                "v_max (angstrom)": float(context.v_edges[v_bin + 1]),
                "v_center (angstrom)": float((context.v_edges[v_bin] + context.v_edges[v_bin + 1]) / 2.0),
                "defined_center_count": int(aggregates[0][index]),
                "positive_count": int(aggregates[1][index]),
                "negative_count": int(aggregates[2][index]),
                "zero_count": int(aggregates[3][index]),
                "positive_fraction": float(aggregates[5][index]),
                "negative_fraction": float(aggregates[6][index]),
                "zero_fraction": float(aggregates[7][index]),
                "mean_polarity": float(aggregates[4][index]),
                "bin_reference_frame": context.reference_frame,
            })

    profile_count = len(context.profile_edges) - 1
    profile_aggregates = _aggregate(signs, profile_bins, profile_count)
    kymograph_rows = []
    for profile_bin in range(profile_count):
        kymograph_rows.append({
            "frame_index": payload.frame_index,
            "iter": payload.iteration,
            "profile_axis": request.profile_axis,
            "component": request.component,
            "profile_bin": profile_bin,
            "coordinate_min (angstrom)": float(context.profile_edges[profile_bin]),
            "coordinate_max (angstrom)": float(context.profile_edges[profile_bin + 1]),
            "coordinate_center (angstrom)": float((context.profile_edges[profile_bin] + context.profile_edges[profile_bin + 1]) / 2.0),
            "defined_center_count": int(profile_aggregates[0][profile_bin]),
            "positive_count": int(profile_aggregates[1][profile_bin]),
            "negative_count": int(profile_aggregates[2][profile_bin]),
            "zero_count": int(profile_aggregates[3][profile_bin]),
            "positive_fraction": float(profile_aggregates[5][profile_bin]),
            "negative_fraction": float(profile_aggregates[6][profile_bin]),
            "zero_fraction": float(profile_aggregates[7][profile_bin]),
            "mean_polarity": float(profile_aggregates[4][profile_bin]),
            "bin_reference_frame": context.reference_frame,
        })
    return center_rows, projected_rows, kymograph_rows


def _run_payloads(
    payloads: Iterable[_FramePayload],
    request,
    context,
    reporter=None,
    pipeline: BoundedFramePipeline | None = None,
):
    writer = getattr(pipeline, "artifact_writer", None) if pipeline is not None else None
    center_sink = (
        writer.sink("centers")
        if writer is not None and request.include_centers and "centers" in writer.specs
        else None
    )
    centers = TableAccumulator(retain=center_sink is None, sink=center_sink)
    projected = TableAccumulator()
    kymograph = TableAccumulator()
    frame_indices, iterations = [], []

    def consume(payload, result):
        center_rows, projected_rows, kymograph_rows = result
        centers.add(center_rows)
        projected.add(projected_rows)
        kymograph.add(kymograph_rows)
        frame_indices.append(payload.frame_index)
        iterations.append(payload.iteration)
        if callable(reporter):
            reporter("analyze", len(frame_indices), 0, "Analyzing basal-plane projected polarity frames")

    if pipeline is None:
        policy = resolve_execution_policy(_PipelineContract(), request, {})
        pipeline = BoundedFramePipeline(policy)
    for completed in pipeline.map_reference(
        payloads,
        lambda: context,
        lambda item, state: _frame_rows(item, request, state),
    ):
        consume(completed.envelope.payload, completed.value)

    return BasalPlaneProjectedPolarityResult(
        centers=centers.finalize(),
        projected_bins=projected.finalize(),
        kymograph_bins=kymograph.finalize(),
        request=request,
        dipole_result=None,
        frame_indices=np.asarray(frame_indices, dtype=int),
        iterations=np.asarray(iterations, dtype=int),
        projection_edges=(context.u_edges, context.v_edges),
        profile_edges=context.profile_edges,
    )


def calculate_basal_plane_projected_polarity(data, request, reporter=None):
    """Calculate projected signs directly, without all-ion intermediate tables."""
    _validate_request(request)
    if request.profile_axis is None:
        request.profile_axis = cast(CartesianAxis, request.projection_plane[1])
    trajectory, _ = _trajectory_and_charges(data)
    selected = _selected_frames(trajectory, request)
    reference_frame = int(request.reference_frame)
    if reference_frame < 0 or reference_frame >= len(trajectory.positions):
        raise ValueError(f"Reference frame {reference_frame} is not present in the trajectory.")
    reference_payload = _payload(data, reference_frame, _source_frame(trajectory, reference_frame), request)
    context = _build_context(reference_payload, request)
    payloads = (
        _payload(data, frame, _source_frame(trajectory, frame), request)
        for frame in selected
    )
    return _run_payloads(payloads, request, context, reporter=reporter)


@register_task(
    "get-basal-plane-displacement-projected-polarity",
    label="Basal-plane Projected Polarity",
)
class BasalPlaneProjectedPolarityTask(AnalysisTask):
    """Stream and bin basal-plane center signs without all-ion outputs."""

    required_data = TrajectoryData
    supports_selective_streaming = True
    native_charges_required_for_auto = True
    execution_capabilities = _PIPELINE_CAPABILITIES
    VERSION = "3"

    def required_data_for(self, request, args: dict | None = None):
        return required_wurtzite_data_type(request, args)

    @staticmethod
    def required_data_fields_for(request, _args: dict) -> tuple[str, ...]:
        return ("trajectory", "charges") if request.charge_source != "formal" else ("trajectory",)

    @staticmethod
    def recommended_presentations(_result, _payload: dict[str, Any]) -> list[PresentationSpec]:
        return [PresentationSpec(renderer="table", label="Projected mean polarity", view_type="table")]

    def run(self, data, request, reporter=None):
        return calculate_basal_plane_projected_polarity(data, request, reporter=reporter)

    def run_stream(self, frames, request, reporter=None, pipeline=None):
        _validate_request(request)
        if request.profile_axis is None:
            request.profile_axis = cast(CartesianAxis, request.projection_plane[1])
        pipeline = pipeline or BoundedFramePipeline(resolve_execution_policy(self, request, {}))
        with reference_frames(frames, request.reference_frame) as (reference_data, selected_source):
            reference_payload = _payload(reference_data, 0, int(request.reference_frame), request)
            reference_request = request
            with pipeline.measure_stage("pipeline_prepare", reference_frame=int(request.reference_frame)):
                context = _build_context(reference_payload, request)
            stride = max(1, int(request.every))
            wanted = None if request.frames is None else set(list(request.frames)[::stride])

            def selected_payloads():
                for occurrence, (source, data) in enumerate(selected_source):
                    if wanted is not None:
                        if source not in wanted:
                            continue
                    elif occurrence % stride:
                        continue
                    yield _payload(data, 0, source, reference_request)

            return _run_payloads(selected_payloads(), request, context, reporter=reporter, pipeline=pipeline)


__all__ = [
    "BasalPlaneProjectedPolarityRequest",
    "BasalPlaneProjectedPolarityResult",
    "BasalPlaneProjectedPolarityTask",
    "CartesianAxis",
    "ProjectionPlane",
    "calculate_basal_plane_projected_polarity",
]
