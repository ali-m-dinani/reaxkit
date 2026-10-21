"""Projected signs of Al displacement relative to coplanar reference N neighbors."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass, field as dc_field
from typing import Any, ClassVar, Iterable, Literal, cast

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.neighbors import (
    _trajectory_and_charges,
)
from reaxkit.analysis.ferroelectrics.hbn_reference.local_polarization import (
    HBNReferenceLocalPolarizationRequest,
    HBNReferenceLocalPolarizationResult,
)
from reaxkit.analysis.ferroelectrics.hbn_reference.polarization import (
    PreparedHBNReference,
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
from reaxkit.analysis.ferroelectrics.hbn_reference.al_neighborhoods import AlNeighborhoods, build_al_neighborhoods
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
from reaxkit.domain.data_models import ElectrostaticsData, TrajectoryData
from reaxkit.presentation.specs import PresentationSpec
from reaxkit.engine.common.generators.extended_xyz_generator import ExtendedXYZFrame, ExtendedXYZWriter

ProjectionPlane = Literal["xy", "xz", "yz"]
PolarityComponent = Literal["x", "y", "z", "c"]
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
    dipole_zero_tolerance: float | None = None
    include_centers: bool = True
    write_extxyz: bool = False
    extxyz_precision: int = 8
    _output_path: str | None = dc_field(default=None, repr=False, compare=False)
    workers: int = 0
    chunk_size: int = 0


@dataclass
class HBNReferenceProjectedPolarityResult(BaseResult):
    """Local-group signs plus projected and frame-resolved mean-polarity tables."""

    time_unit = "ps"

    artifact_tiers = {
        "hbn_reference_projected_polarity_2d": "core",
        "hbn_reference_projected_polarity_kymograph": "core",
        "hbn_reference_projected_polarity_whole_slab_summary": "summary",
        "hbn_reference_projected_polarity_centers": "detail",
    }

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
    neighborhoods: AlNeighborhoods
    atom_ids: np.ndarray
    reference_centers: np.ndarray
    u_edges: np.ndarray
    v_edges: np.ndarray
    u_bins: np.ndarray
    v_bins: np.ndarray
    profile_bins: np.ndarray
    profile_edges: np.ndarray
    c_hat: np.ndarray


@dataclass(frozen=True)
class _FramePayload:
    frame_index: int
    iteration: int
    xyz: np.ndarray
    labels: np.ndarray
    cell: np.ndarray
    charges: np.ndarray | None
    time_ps: float


def _validate_request(request: HBNReferenceProjectedPolarityRequest) -> None:
    if request.component not in {"x", "y", "z", "c"}:
        raise ValueError("component must be 'x', 'y', 'z', or 'c'.")
    if request.projection_plane not in {"xy", "xz", "yz"}:
        raise ValueError("projection_plane must be 'xy', 'xz', or 'yz'.")
    if request.profile_axis is not None and request.profile_axis not in request.projection_plane:
        raise ValueError("profile_axis must be one of the two projection-plane axes.")
    if len(request.projection_bins) != 2 or any(int(value) <= 0 for value in request.projection_bins):
        raise ValueError("projection_bins must contain two positive integers.")
    if request.dipole_zero_tolerance is not None:
        raise ValueError("Projected polarity is geometric; use geometric_zero_tolerance in angstrom.")
    if not np.isfinite(request.geometric_zero_tolerance) or request.geometric_zero_tolerance < 0:
        raise ValueError("geometric_zero_tolerance must be finite and non-negative.")
    if request.local_grouping not in {"al", "layer"}:
        raise ValueError("Geometric projected polarity is per Al; use local_grouping='al'.")
    if request.local_charge_treatment is not None:
        raise ValueError("Geometric projected polarity does not use local_charge_treatment.")
    if request.write_extxyz and not request._output_path:
        raise ValueError("EXTXYZ export requires an output path.")
    if int(request.extxyz_precision) < 1:
        raise ValueError("extxyz_precision must be at least 1.")
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
    neighborhoods = build_al_neighborhoods(prepared)
    reference_centers = neighborhoods.reference_centers
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
        neighborhoods=neighborhoods,
        atom_ids=np.asarray(trajectory.atom_ids, dtype=int),
        reference_centers=np.asarray(reference_centers, dtype=float),
        u_edges=u_edges,
        v_edges=v_edges,
        u_bins=u_bins,
        v_bins=v_bins,
        profile_bins=profile_bins,
        profile_edges=profile_edges,
        c_hat=_unit_vector(request.c_axis, "c_axis"),
    )


def _payload(data, frame: int, source_frame: int, request) -> _FramePayload:
    trajectory, dynamic_charges = _trajectory_and_charges(data)
    from reaxkit.core.results_shaping.result_time_enrichment import _attach_time_to_frame, _iter_time_mapping_from_data
    timing = _attach_time_to_frame(
        pd.DataFrame({"iter": [_iteration(trajectory, frame)]}),
        iter_to_time=_iter_time_mapping_from_data(data),
        control_file=getattr(request, "_control_file", "control"), time_unit="ps",
    )
    return _FramePayload(
        frame_index=int(source_frame),
        iteration=_iteration(trajectory, frame),
        xyz=np.asarray(trajectory.positions[frame], dtype=float),
        labels=_frame_labels(trajectory, frame),
        cell=_frame_cell(trajectory, request, frame),
        charges=None,
        time_ps=float(timing["time"].iloc[0]) if "time" in timing else np.nan,
    )




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
    relative_xyz = context.neighborhoods.relative_displacement(displacement, payload.cell, request.periodic)
    relative = (
        relative_xyz[:, "xyz".index(request.component)]
        if request.component in "xyz"
        else relative_xyz @ context.c_hat
    )
    neighborhoods = context.neighborhoods
    group_count = len(context.reference_centers)
    polarity = np.sign(relative)
    polarity[np.abs(relative) <= float(request.geometric_zero_tolerance)] = 0.0
    polarity[~np.isfinite(relative)] = np.nan

    center_rows = []
    if request.include_centers:
        current_centers = context.reference_centers + displacement[neighborhoods.al_indices]
        for group_id in range(group_count):
            row = {
                "frame_index": payload.frame_index,
                "iter": payload.iteration,
                "local_neighborhood_id": int(neighborhoods.layer_ids[group_id]),
                "local_grouping": "al",
                "representative_atom_id": int(context.atom_ids[neighborhoods.al_indices[group_id]]),
                "neighbor_count": 3,
                "polarity_method": "geometric",
                f"geometric_displacement_{request.component} (angstrom)": float(relative[group_id]),
                "geometric_zero_tolerance (angstrom)": float(request.geometric_zero_tolerance),
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
        "local_grouping": "al",
        "polarity_method": "geometric",
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
                "local_grouping": "al",
                "polarity_method": "geometric",
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
            "local_grouping": "al",
            "polarity_method": "geometric",
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
    for row in [*center_rows, *projected_rows, *kymograph_rows, slab_row]:
        row["time"] = payload.time_ps
        row["time_unit"] = "ps"
    extended = None
    if request.write_extxyz:
        count = len(xyz)
        centers = neighborhoods.al_indices
        ids = np.full(count, -1, dtype=int)
        ids[centers] = neighborhoods.layer_ids
        is_center = np.zeros(count, dtype=int)
        is_center[centers] = 1
        properties = {"atom_number": context.atom_ids, "local_neighborhood_id": ids, "is_local_center": is_center}
        relative_c = relative_xyz @ context.c_hat
        for name, values in {
            "geometric_displacement": relative_xyz,
            "geometric_displacement_c": relative_c,
            "geometric_polarity": np.where(np.abs(relative_xyz) <= request.geometric_zero_tolerance, 0.0, np.sign(relative_xyz)),
            "geometric_polarity_c": np.where(np.abs(relative_c) <= request.geometric_zero_tolerance, 0.0, np.sign(relative_c)),
            "polarity": polarity,
        }.items():
            array = np.full((count, *values.shape[1:]), np.nan)
            array[centers] = values
            properties[name] = array
        extended = ExtendedXYZFrame(
            species=payload.labels.astype(str).tolist(), positions=xyz, properties=properties,
            frame=payload.frame_index, iteration=payload.iteration, lattice=payload.cell,
            pbc=tuple(bool(value) for value in request.periodic),
            metadata={"polarity_method": "geometric", "polarity_component": request.component,
                      "time": payload.time_ps, "time_unit": "ps",
                      "local_grouping": "neighborhood", "geometric_displacement_units": "angstrom",
                      "geometric_zero_tolerance": request.geometric_zero_tolerance,
                      "geometric_polarity_convention": "Al_minus_mean_N_displacement"},
        )
    return center_rows, projected_rows, kymograph_rows, slab_row, extended


def _finish_checkpoint(durable, request, store):
    result = durable.finish(request)
    if request.write_extxyz:
        from reaxkit.core.runtime.checkpoint_trajectory import publish_segments
        publish_segments(store, request._output_path)
    result.skip_result_cache = True
    return result


def _run_payloads(
    payloads: Iterable[_FramePayload],
    request,
    context,
    reporter=None,
    pipeline: BoundedFramePipeline | None = None,
):
    from reaxkit.core.runtime.checkpoint_results import CheckpointAccumulator
    store = getattr(request, "_result_store", None)
    durable = CheckpointAccumulator(store) if store is not None else None
    if store is not None and store.manifest["state"] in {"analysis_complete", "complete"}:
        return _finish_checkpoint(durable, request, store)
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
        center_rows, projected_rows, kymograph_rows, slab_row, extended = result
        if durable is not None:
            position = store.committed_frames + len(store.pending)
            frame_result = HBNReferenceProjectedPolarityResult(
                centers=pd.DataFrame(center_rows), projected_bins=pd.DataFrame(projected_rows),
                kymograph_bins=pd.DataFrame(kymograph_rows), whole_slab_summary=pd.DataFrame([slab_row]),
                request=request, local_result=None,
                frame_indices=np.asarray([int(slab_row["frame_index"])]), iterations=np.asarray([int(slab_row["iter"])]),
                projection_edges=(context.u_edges, context.v_edges), profile_edges=context.profile_edges)
            extra = {}
            if extended is not None:
                from reaxkit.core.runtime.checkpoint_trajectory import segment_table
                extra["_extxyz"] = segment_table(extended, request.extxyz_precision)
            durable.add(frame_result, position, int(slab_row["frame_index"]), extra_tables=extra)
            return
        if extended is not None:
            xyz_writer.write_frame(extended)
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
    selected_payloads = (item for position, item in enumerate(payloads) if store is None or position >= store.committed_frames)
    xyz_context = (
        ExtendedXYZWriter(request._output_path, precision=request.extxyz_precision)
        if request.write_extxyz and store is None else nullcontext()
    )
    with xyz_context as xyz_writer:
        for completed in pipeline.map_reference(
            selected_payloads,
            lambda: context,
            lambda item, state: _frame_rows(item, request, state),
        ):
            consume(completed.value)

    if durable is not None:
        return _finish_checkpoint(durable, request, store)

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
    native_charges_required_for_auto = False
    execution_capabilities = _PIPELINE_CAPABILITIES
    VERSION = "8"

    def required_data_for(self, request, args: dict | None = None):
        return TrajectoryData

    @staticmethod
    def required_data_fields_for(request, args: dict) -> tuple[str, ...]:
        return ("trajectory",)

    @staticmethod
    def recommended_presentations(_result, _payload: dict[str, Any]) -> list[PresentationSpec]:
        return [PresentationSpec(renderer="table", label="h-BN-reference projected mean polarity", view_type="table")]

    def run(self, data, request, reporter=None):
        return calculate_hbn_reference_projected_polarity(data, request, reporter=reporter)

    def run_stream(self, frames, request, reporter=None, pipeline=None):
        store = getattr(request, "_result_store", None)
        if store is not None and store.manifest["state"] in {"analysis_complete", "complete"}:
            from reaxkit.core.runtime.checkpoint_results import CheckpointAccumulator
            return _finish_checkpoint(CheckpointAccumulator(store), request, store)
        _validate_request(request)
        if request.profile_axis is None:
            request.profile_axis = cast(CartesianAxis, request.projection_plane[1])
        pipeline = pipeline or BoundedFramePipeline(resolve_execution_policy(self, request, {}))
        with reference_frames(frames, request.reference_frame) as (reference_data, selected_source):
            reference_trajectory, _ = _trajectory_and_charges(reference_data)
            reference_request = copy.copy(request)
            reference_request.reference_frame = 0
            reference_request.frames = (0,)
            reference_request.every = 1
            with pipeline.measure_stage("pipeline_prepare", reference_frame=int(request.reference_frame)):
                context = _build_context(reference_trajectory, reference_request)
            stride = max(1, int(request.every))
            wanted = None if request.frames is None else set(list(request.frames)[::stride])

            def selected_payloads():
                seen = set()
                for occurrence, (source, data) in enumerate(selected_source):
                    if wanted is not None:
                        if source not in wanted:
                            continue
                    elif occurrence % stride:
                        continue
                    seen.add(source)
                    yield _payload(data, 0, source, reference_request)
                if wanted is not None and wanted - seen:
                    raise ValueError(f"Requested frames not found: {sorted(wanted - seen)}")

            return _run_payloads(selected_payloads(), request, context, reporter=reporter, pipeline=pipeline)


__all__ = [
    "CartesianAxis",
    "HBNReferenceProjectedPolarityRequest",
    "HBNReferenceProjectedPolarityResult",
    "HBNReferenceProjectedPolarityTask",
    "PolarityComponent",
    "ProjectionPlane",
    "calculate_hbn_reference_projected_polarity",
]
