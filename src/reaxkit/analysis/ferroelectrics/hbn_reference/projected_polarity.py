"""Projected polarity fractions from local h-BN-reference group dipoles."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field as dc_field
from typing import Any, ClassVar, Iterable, Literal, cast

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.neighbors import (
    _trajectory_and_charges,
    required_wurtzite_data_type,
)
from reaxkit.analysis.ferroelectrics.hbn_reference.local_polarization import (
    HBNReferenceLocalPolarizationRequest,
    HBNReferenceLocalPolarizationResult,
)
from reaxkit.analysis.ferroelectrics.hbn_reference.polarization import (
    ELECTRON_CHARGE_SIGN,
    PreparedHBNReference,
    _formal_charge_map,
    _frame_cell,
    _frame_labels,
    _iteration,
    _minimum_image,
    _periodic_axes,
    _selected_frames,
    _translation_from_assignment,
    _unit_vector,
    prepare_hbn_reference,
)
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.core.runtime.execution_contracts import (
    ExecutionShape,
    TaskCapabilities,
    resolve_execution_policy,
)
from reaxkit.core.runtime.frame_pipeline import BoundedFramePipeline
from reaxkit.core.runtime.reducers import TableAccumulator
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ElectrostaticsData, TrajectoryData
from reaxkit.presentation.specs import PresentationSpec

ProjectionPlane = Literal["xy", "xz", "yz"]
PolarityComponent = Literal["x", "y", "z", "c"]
CartesianAxis = Literal["x", "y", "z"]

_PIPELINE_CAPABILITIES = TaskCapabilities(
    shape=ExecutionShape.REFERENCE_FRAME_MAP,
    thread_safe=True,
    needs_reference=True,
    estimated_frame_bytes=16 * 1024 * 1024,
)


class _PipelineContract:
    execution_capabilities = _PIPELINE_CAPABILITIES


@dataclass
class HBNReferenceProjectedPolarityRequest(HBNReferenceLocalPolarizationRequest):
    """Configure fixed-reference spatial averaging of local-group signs."""

    stream_reference_first: ClassVar[bool] = True
    component: PolarityComponent = dc_field(
        default="c", metadata={"label": "Polarity component", "choices": ["x", "y", "z", "c"]}
    )
    projection_plane: ProjectionPlane = dc_field(
        default="xz", metadata={"label": "Projection plane", "choices": ["xy", "xz", "yz"]}
    )
    projection_bins: tuple[int, int] = (40, 40)
    profile_axis: CartesianAxis | None = dc_field(
        default=None,
        metadata={"label": "Kymograph profile axis", "choices": ["x", "y", "z"]},
    )
    dipole_zero_tolerance: float = 0.0
    include_centers: bool = True
    workers: int = 0
    chunk_size: int = 0


@dataclass
class HBNReferenceProjectedPolarityResult(BaseResult):
    """Local-group signs plus projected and frame-resolved mean-polarity tables."""

    centers: pd.DataFrame
    projected_bins: pd.DataFrame
    kymograph_bins: pd.DataFrame
    whole_slab_summary: pd.DataFrame
    request: HBNReferenceProjectedPolarityRequest
    local_result: HBNReferenceLocalPolarizationResult | None
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
            "hbn_reference_projected_polarity_2d": self.projected_bins,
            "hbn_reference_projected_polarity_kymograph": self.kymograph_bins,
            "hbn_reference_projected_polarity_whole_slab_summary": self.whole_slab_summary,
        }
        if not self.centers.empty:
            tables["hbn_reference_projected_polarity_centers"] = self.centers
        return tables


@dataclass(frozen=True)
class _ProjectionContext:
    prepared: PreparedHBNReference
    atom_group_ids: np.ndarray
    reference_centers: np.ndarray
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
    cell: np.ndarray
    charges: np.ndarray | None


def _validate_request(request: HBNReferenceProjectedPolarityRequest) -> None:
    if request.component not in {"x", "y", "z", "c"}:
        raise ValueError("component must be 'x', 'y', 'z', or 'c'.")
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


def _polarity_statistics(values: np.ndarray):
    signs = np.asarray(values, dtype=float)
    signs = signs[np.isfinite(signs)]
    positive = int(np.count_nonzero(signs > 0.0))
    negative = int(np.count_nonzero(signs < 0.0))
    zero = int(np.count_nonzero(signs == 0.0))
    count = positive + negative + zero
    if count == 0:
        return 0, 0, 0, 0, np.nan, np.nan, np.nan, np.nan
    return count, positive, negative, zero, float(np.mean(signs)), positive / count, negative / count, zero / count


def _build_context(trajectory, request) -> _ProjectionContext:
    prepared = prepare_hbn_reference(trajectory, request)
    assignment = prepared.simulation_to_reference
    if request.local_grouping == "cell":
        atom_group_ids = prepared.local_cell_ids[assignment]
        reference_centers = prepared.local_cell_centers
    else:
        atom_group_ids = prepared.local_layer_ids[assignment]
        reference_centers = prepared.local_layer_centers
    u_axis, v_axis = request.projection_plane
    u_reference = reference_centers[:, "xyz".index(u_axis)]
    v_reference = reference_centers[:, "xyz".index(v_axis)]
    u_edges = _edges(u_reference, int(request.projection_bins[0]))
    v_edges = _edges(v_reference, int(request.projection_bins[1]))
    u_bins = _bin_indices(u_reference, u_edges)
    v_bins = _bin_indices(v_reference, v_edges)
    profile_bins, profile_edges = (
        (u_bins, u_edges) if request.profile_axis == u_axis else (v_bins, v_edges)
    )
    return _ProjectionContext(
        prepared=prepared,
        atom_group_ids=np.asarray(atom_group_ids, dtype=int),
        reference_centers=np.asarray(reference_centers, dtype=float),
        u_edges=u_edges,
        v_edges=v_edges,
        u_bins=u_bins,
        v_bins=v_bins,
        profile_bins=profile_bins,
        profile_edges=profile_edges,
        c_hat=_unit_vector(request.c_axis, "c_axis"),
        formal_charges=_formal_charge_map(request.formal_charges),
    )


def _payload(data, frame: int, source_frame: int, request) -> _FramePayload:
    trajectory, dynamic_charges = _trajectory_and_charges(data)
    return _FramePayload(
        frame_index=int(source_frame),
        iteration=_iteration(trajectory, frame),
        xyz=np.asarray(trajectory.positions[frame], dtype=float),
        labels=_frame_labels(trajectory, frame),
        cell=_frame_cell(trajectory, request, frame),
        charges=None if dynamic_charges is None else np.asarray(dynamic_charges[frame], dtype=float),
    )


def _resolve_charges(payload, request, context) -> np.ndarray:
    use_dynamic = request.charge_source == "reaxff" or (
        request.charge_source == "auto" and payload.charges is not None
    )
    if use_dynamic:
        if payload.charges is None:
            raise ValueError("--charge-source reaxff requires per-atom charges from fort.7.")
        charges = np.asarray(payload.charges, dtype=float)
    else:
        charges = np.asarray(
            [context.formal_charges.get(str(label).casefold(), np.nan) for label in payload.labels]
        )
    if not np.isfinite(charges).all():
        bad = sorted({str(payload.labels[index]) for index in np.flatnonzero(~np.isfinite(charges))})
        raise ValueError(
            f"Frame {payload.frame_index} has missing or non-finite atomic charge(s): {', '.join(bad)}."
        )
    return charges


def _sum_by_group(values, groups, count) -> np.ndarray:
    return np.bincount(groups, weights=np.asarray(values, dtype=float), minlength=count)


def _frame_rows(payload, request, context):
    xyz = payload.xyz
    prepared = context.prepared
    if not np.isfinite(xyz).all():
        raise ValueError(f"Frame {payload.frame_index} contains non-finite coordinates.")
    if len(xyz) != len(prepared.simulation_to_reference):
        raise ValueError(f"Frame {payload.frame_index} has a different atom count.")
    fractional = xyz @ np.linalg.inv(payload.cell)
    translation = _translation_from_assignment(
        fractional,
        prepared.fractional_positions,
        prepared.simulation_to_reference,
        payload.cell,
        _periodic_axes(request.periodic),
    )
    reference_for_atoms = prepared.fractional_positions[prepared.simulation_to_reference] + translation
    displacement = _minimum_image(
        fractional - reference_for_atoms, _periodic_axes(request.periodic)
    ) @ payload.cell
    component_displacement = (
        displacement[:, "xyz".index(request.component)]
        if request.component in "xyz"
        else displacement @ context.c_hat
    )
    charges = _resolve_charges(payload, request, context)
    groups = context.atom_group_ids
    group_count = len(context.reference_centers)
    atom_count = np.bincount(groups, minlength=group_count).astype(float)
    net_charge = _sum_by_group(charges, groups, group_count)
    displacement_sum = _sum_by_group(component_displacement, groups, group_count)
    raw_dipole = _sum_by_group(
        ELECTRON_CHARGE_SIGN * charges * component_displacement, groups, group_count
    )
    neutralize = np.full(group_count, request.local_charge_treatment == "neutralize")
    if request.local_charge_treatment == "auto":
        neutralize = np.abs(net_charge) > 1.0e-10
    neutralized = raw_dipole - ELECTRON_CHARGE_SIGN * net_charge / atom_count * displacement_sum
    dipole = np.where(neutralize, neutralized, raw_dipole)
    polarity = np.sign(dipole)
    polarity[np.abs(dipole) <= float(request.dipole_zero_tolerance)] = 0.0
    polarity[~np.isfinite(dipole)] = np.nan

    center_rows = []
    if request.include_centers:
        mean_displacement = np.column_stack(
            [_sum_by_group(displacement[:, axis], groups, group_count) / atom_count for axis in range(3)]
        )
        current_centers = context.reference_centers + mean_displacement
        group_column = "local_cell_id" if request.local_grouping == "cell" else "local_layer_id"
        for group_id in range(group_count):
            row = {
                "frame_index": payload.frame_index,
                "iter": payload.iteration,
                group_column: group_id,
                "local_group_id": group_id,
                "local_grouping": request.local_grouping,
                "atom_count": int(atom_count[group_id]),
                "net_charge (e)": float(net_charge[group_id]),
                "charge_neutralization_applied": bool(neutralize[group_id]),
                f"raw_dipole_{request.component} (e*angstrom)": float(raw_dipole[group_id]),
                f"dipole_{request.component} (e*angstrom)": float(dipole[group_id]),
                "polarity_component": request.component,
                "has_defined_polarity": bool(np.isfinite(polarity[group_id])),
                "polarity": float(polarity[group_id]),
                "reference_u (angstrom)": float(context.reference_centers[group_id, "xyz".index(request.projection_plane[0])]),
                "reference_v (angstrom)": float(context.reference_centers[group_id, "xyz".index(request.projection_plane[1])]),
                "u_bin": int(context.u_bins[group_id]),
                "v_bin": int(context.v_bins[group_id]),
                "bin_reference_frame": int(request.reference_frame),
            }
            for axis, name in enumerate("xyz"):
                row[f"reference_center_{name} (angstrom)"] = float(context.reference_centers[group_id, axis])
                row[f"center_{name} (angstrom)"] = float(current_centers[group_id, axis])
            center_rows.append(row)

    stats = _polarity_statistics(polarity)
    slab_row = {
        "frame_index": payload.frame_index,
        "iter": payload.iteration,
        "component": request.component,
        "local_grouping": request.local_grouping,
        "spatial_scope": "whole_slab",
        "total_group_count": group_count,
        "defined_group_count": stats[0],
        "undefined_group_count": group_count - stats[0],
        "positive_count": stats[1],
        "negative_count": stats[2],
        "zero_count": stats[3],
        "positive_fraction": stats[5],
        "negative_fraction": stats[6],
        "zero_fraction": stats[7],
        "positive_percentage": 100.0 * stats[5],
        "negative_percentage": 100.0 * stats[6],
        "zero_percentage": 100.0 * stats[7],
        "mean_polarity": stats[4],
        "bin_reference_frame": int(request.reference_frame),
    }

    projected_rows = []
    u_axis, v_axis = request.projection_plane
    nu, nv = (int(value) for value in request.projection_bins)
    for v_bin in range(nv):
        for u_bin in range(nu):
            stats = _polarity_statistics(
                polarity[(context.u_bins == u_bin) & (context.v_bins == v_bin)]
            )
            projected_rows.append({
                "frame_index": payload.frame_index,
                "iter": payload.iteration,
                "plane": request.projection_plane,
                "component": request.component,
                "local_grouping": request.local_grouping,
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
                "defined_cell_count": stats[0],
                "defined_group_count": stats[0],
                "positive_count": stats[1],
                "negative_count": stats[2],
                "zero_count": stats[3],
                "positive_fraction": stats[5],
                "negative_fraction": stats[6],
                "zero_fraction": stats[7],
                "mean_polarity": stats[4],
                "bin_reference_frame": int(request.reference_frame),
            })

    kymograph_rows = []
    for profile_bin in range(len(context.profile_edges) - 1):
        stats = _polarity_statistics(polarity[context.profile_bins == profile_bin])
        kymograph_rows.append({
            "frame_index": payload.frame_index,
            "iter": payload.iteration,
            "profile_axis": request.profile_axis,
            "component": request.component,
            "local_grouping": request.local_grouping,
            "profile_bin": profile_bin,
            "coordinate_min (angstrom)": float(context.profile_edges[profile_bin]),
            "coordinate_max (angstrom)": float(context.profile_edges[profile_bin + 1]),
            "coordinate_center (angstrom)": float((context.profile_edges[profile_bin] + context.profile_edges[profile_bin + 1]) / 2.0),
            "defined_cell_count": stats[0],
            "defined_group_count": stats[0],
            "positive_count": stats[1],
            "negative_count": stats[2],
            "zero_count": stats[3],
            "positive_fraction": stats[5],
            "negative_fraction": stats[6],
            "zero_fraction": stats[7],
            "mean_polarity": stats[4],
            "bin_reference_frame": int(request.reference_frame),
        })
    return center_rows, projected_rows, kymograph_rows, slab_row


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
    slab = TableAccumulator()
    frame_indices, iterations = [], []

    def consume(result):
        center_rows, projected_rows, kymograph_rows, slab_row = result
        centers.add(center_rows)
        projected.add(projected_rows)
        kymograph.add(kymograph_rows)
        slab.add([slab_row])
        frame_indices.append(int(slab_row["frame_index"]))
        iterations.append(int(slab_row["iter"]))
        if callable(reporter):
            reporter("analyze", len(frame_indices), 0, "Analyzing projected polarity frames")

    if pipeline is None:
        policy = resolve_execution_policy(_PipelineContract(), request, {})
        pipeline = BoundedFramePipeline(policy)
    for completed in pipeline.map_ordered(
        payloads,
        lambda item: _frame_rows(item, request, context),
    ):
        consume(completed.value)

    return HBNReferenceProjectedPolarityResult(
        centers=centers.finalize(),
        projected_bins=projected.finalize(),
        kymograph_bins=kymograph.finalize(),
        whole_slab_summary=slab.finalize(),
        request=request,
        local_result=None,
        frame_indices=np.asarray(frame_indices, dtype=int),
        iterations=np.asarray(iterations, dtype=int),
        projection_edges=(context.u_edges, context.v_edges),
        profile_edges=context.profile_edges,
    )


def calculate_hbn_reference_projected_polarity(data, request, reporter=None):
    """Average local polarities without materializing per-atom displacement tables."""
    _validate_request(request)
    if request.profile_axis is None:
        request.profile_axis = cast(CartesianAxis, request.projection_plane[1])
    trajectory, _ = _trajectory_and_charges(data)
    selected = _selected_frames(trajectory, request)
    context = _build_context(trajectory, request)
    payloads = (_payload(data, frame, frame, request) for frame in selected)
    return _run_payloads(payloads, request, context, reporter=reporter)


def _source_frame(data, fallback: int) -> int:
    trajectory, _ = _trajectory_and_charges(data)
    values = trajectory.source_frame_indices
    return int(np.asarray(values).reshape(-1)[0]) if values is not None else int(fallback)


@register_task("get-hbn-reference-projected-polarity", label="h-BN-reference Projected Polarity")
class HBNReferenceProjectedPolarityTask(AnalysisTask):
    """Bin local reference-group signs with bounded-memory frame processing."""

    required_data = TrajectoryData
    supports_selective_streaming = True
    native_charges_required_for_auto = True
    execution_capabilities = _PIPELINE_CAPABILITIES
    VERSION = "3"

    def required_data_for(self, request, args: dict | None = None):
        return required_wurtzite_data_type(request, args)

    @staticmethod
    def required_data_fields_for(request, args: dict) -> tuple[str, ...]:
        return (("trajectory",) if required_wurtzite_data_type(request, args) is TrajectoryData else ("trajectory", "charges"))

    @staticmethod
    def recommended_presentations(_result, _payload: dict[str, Any]) -> list[PresentationSpec]:
        return [PresentationSpec(renderer="table", label="h-BN-reference projected mean polarity", view_type="table")]

    def run(self, data, request, reporter=None):
        return calculate_hbn_reference_projected_polarity(data, request, reporter=reporter)

    def run_stream(self, frames, request, reporter=None, pipeline=None):
        _validate_request(request)
        if request.profile_axis is None:
            request.profile_axis = cast(CartesianAxis, request.projection_plane[1])
        iterator = iter(frames)
        buffered = []
        reference_data = None
        for stream_index, data in enumerate(iterator):
            source = _source_frame(data, stream_index)
            buffered.append((source, data))
            if source == int(request.reference_frame):
                reference_data = data
                break
        if reference_data is None:
            raise ValueError(f"Reference frame {request.reference_frame} was not present in the input stream.")
        reference_trajectory, _ = _trajectory_and_charges(reference_data)
        reference_request = copy.copy(request)
        reference_request.reference_frame = 0
        reference_request.frames = (0,)
        reference_request.every = 1
        if pipeline is None:
            pipeline = BoundedFramePipeline(
                resolve_execution_policy(self, request, {})
            )
        with pipeline.measure_stage(
            "pipeline_prepare", reference_frame=int(request.reference_frame)
        ):
            context = _build_context(reference_trajectory, reference_request)
        wanted = None if request.frames is None else {int(value) for value in request.frames}

        def selected_payloads():
            try:
                occurrence = 0
                for source, data in buffered:
                    if wanted is not None and source not in wanted:
                        continue
                    if occurrence % max(1, int(request.every)) == 0:
                        yield _payload(data, 0, source, reference_request)
                    occurrence += 1
                for stream_index, data in enumerate(iterator, start=len(buffered)):
                    source = _source_frame(data, stream_index)
                    if wanted is not None and source not in wanted:
                        continue
                    if occurrence % max(1, int(request.every)) == 0:
                        yield _payload(data, 0, source, reference_request)
                    occurrence += 1
            finally:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()

        result = _run_payloads(
            selected_payloads(),
            request,
            context,
            reporter=reporter,
            pipeline=pipeline,
        )
        result.request = request
        return result


__all__ = [
    "CartesianAxis",
    "HBNReferenceProjectedPolarityRequest",
    "HBNReferenceProjectedPolarityResult",
    "HBNReferenceProjectedPolarityTask",
    "PolarityComponent",
    "ProjectionPlane",
    "calculate_hbn_reference_projected_polarity",
]
