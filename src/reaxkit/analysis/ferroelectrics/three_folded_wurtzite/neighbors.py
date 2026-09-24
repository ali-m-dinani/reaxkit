"""Neighbor geometry for three-basal-neighbor wurtzite surface sites."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from reaxkit.core.runtime.execution_contracts import TaskCapabilities, ExecutionShape
from reaxkit.analysis.ferroelectrics.neighbor_stream import stream_neighbors
from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.neighbors import (
    WurtziteNeighborRequest as _FourFoldNeighborRequest,
    _charges_for_request,
    _formal_charge_map,
    _frame_cell,
    _frame_iteration,
    _frame_labels,
    _has_neighbor_within_cutoff,
    _periodic_images,
    _select_frames,
    _source_frame,
    _validate_request,
    centers_csv_table,
    neighbors_csv_table,
    required_wurtzite_data_type,
    unit_vector,
)
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import TrajectoryData
from reaxkit.presentation.specs import PresentationSpec


CENTER_COLUMNS = [
    "frame_index", "iter", "site_atom_index", "site_atom_id", "site_element",
    "site_x (angstrom)", "site_y (angstrom)", "site_z (angstrom)",
    "site_charge (e)", "neighbor_count_within_cutoff", "has_three_basal_neighbors",
    "has_apical_neighbor", "has_proton_within_cutoff", "proton_cutoff (angstrom)",
    "charge_source", "neighbor_cutoff (angstrom)", "c_axis_x", "c_axis_y", "c_axis_z",
    "periodic_a", "periodic_b", "periodic_c",
]

NEIGHBOR_COLUMNS = [
    "frame_index", "iter", "site_atom_id", "site_element", "neighbor_rank",
    "neighbor_atom_index", "neighbor_atom_id", "neighbor_element", "neighbor_charge (e)",
    "neighbor_x (angstrom)", "neighbor_y (angstrom)", "neighbor_z (angstrom)",
    "neighbor_image_x (angstrom)", "neighbor_image_y (angstrom)",
    "neighbor_image_z (angstrom)", "distance (angstrom)", "bond_x (angstrom)",
    "bond_y (angstrom)", "bond_z (angstrom)", "bond_c (angstrom)", "neighbor_role",
]


@dataclass
class WurtziteNeighborRequest(_FourFoldNeighborRequest):
    """Configure neighbor extraction for sites with three required basal neighbors."""


@dataclass
class WurtziteNeighborResult(BaseResult):
    """Center rows plus all neighbor candidates inside the distance cutoff."""

    table: pd.DataFrame
    request: WurtziteNeighborRequest
    centers: pd.DataFrame
    neighbors: pd.DataFrame
    frame_indices: np.ndarray
    iterations: np.ndarray

    @property
    def csv_tables(self) -> dict[str, pd.DataFrame]:
        return {
            "centers": centers_csv_table(self.centers),
            "neighbors": neighbors_csv_table(self.neighbors),
        }


def all_neighbors_within_cutoff(
    center_positions: np.ndarray,
    neighbor_positions: np.ndarray,
    *,
    cell: np.ndarray | None,
    periodic: tuple[bool, bool, bool],
    cutoff: float | None,
) -> list[list[tuple[int, float, np.ndarray]]]:
    """Return every unique neighbor, using its nearest periodic image."""

    _, replicas, sources = _periodic_images(neighbor_positions, cell, periodic)
    if cell is not None and any(periodic):
        fractional = np.asarray(center_positions, dtype=float) @ np.linalg.inv(cell)
        for axis, enabled in enumerate(periodic):
            if enabled:
                fractional[:, axis] -= np.floor(fractional[:, axis])
        query_positions = fractional @ cell
    else:
        query_positions = np.asarray(center_positions, dtype=float)

    if cutoff is None:
        candidate_indices = [np.arange(len(replicas), dtype=int)] * len(query_positions)
    else:
        tree = cKDTree(replicas)
        candidate_indices = tree.query_ball_point(
            query_positions,
            r=float(cutoff),
            workers=-1,
        )

    output: list[list[tuple[int, float, np.ndarray]]] = []
    for center, indices in zip(query_positions, candidate_indices, strict=True):
        replica_indices = np.asarray(indices, dtype=int)
        vectors = replicas[replica_indices] - center
        distances = np.linalg.norm(vectors, axis=1)
        nearest_by_source: dict[int, tuple[float, np.ndarray]] = {}
        for local_index, distance in enumerate(distances):
            if not np.isfinite(distance):
                continue
            replica_index = int(replica_indices[local_index])
            source = int(sources[replica_index])
            current = nearest_by_source.get(source)
            if current is None or float(distance) < current[0]:
                nearest_by_source[source] = (float(distance), vectors[local_index].copy())
        output.append([
            (source, distance, vector)
            for source, (distance, vector) in sorted(
                nearest_by_source.items(), key=lambda item: (item[1][0], item[0])
            )
        ])
    return output


def extract_wurtzite_neighbors(
    trajectory: TrajectoryData,
    request: WurtziteNeighborRequest,
    *,
    charges: np.ndarray | None = None,
    frame_indices: Iterable[int] | None = None,
    preserve_source_frame_indices: bool = False,
) -> WurtziteNeighborResult:
    """Find three basal neighbors and retain all other atoms as apical candidates."""

    _validate_request(request)
    positions = np.asarray(trajectory.positions, dtype=float)
    n_frames, n_atoms = positions.shape[:2]
    if charges is not None and np.asarray(charges).shape != (n_frames, n_atoms):
        raise ValueError("charges must have shape (n_frames, n_atoms), matching trajectory positions.")
    charge_array = None if charges is None else np.asarray(charges, dtype=float)
    ids = np.asarray(trajectory.atom_ids, dtype=int)
    selected_frames = (
        _select_frames(n_frames, request)
        if frame_indices is None
        else [int(value) for value in frame_indices]
    )
    c_hat = unit_vector(request.c_axis, "c_axis")
    periodic = tuple(bool(value) for value in request.periodic)
    formal = _formal_charge_map(request)
    center_names = {str(value).casefold() for value in request.centers}
    neighbor_names = {str(value).casefold() for value in request.neighbors}
    proton_names = {str(value).casefold() for value in request.protons}
    center_rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []
    output_frames: list[int] = []
    output_iterations: list[int] = []

    for frame in selected_frames:
        xyz = positions[frame]
        labels = _frame_labels(trajectory, frame)
        folded = np.asarray([str(value).casefold() for value in labels], dtype=object)
        finite = np.isfinite(xyz).all(axis=1)
        center_indices = np.flatnonzero(finite & np.isin(folded, list(center_names)))
        neighbor_indices = np.flatnonzero(finite & np.isin(folded, list(neighbor_names)))
        proton_indices = np.flatnonzero(finite & np.isin(folded, list(proton_names)))
        output_frame = _source_frame(trajectory, frame) if preserve_source_frame_indices else frame
        if center_indices.size == 0:
            raise ValueError(f"Frame {output_frame} contains no selected centers.")
        if neighbor_indices.size < 3:
            raise ValueError(
                f"Frame {output_frame} contains only {neighbor_indices.size} selected neighbor "
                "atoms; at least three are required."
            )
        cell = _frame_cell(trajectory, request, frame)
        frame_periodic = periodic if cell is not None else (False, False, False)
        if any(periodic) and cell is None:
            warnings.warn(
                "Periodic directions were requested but no cell was supplied; using non-periodic distances.",
                RuntimeWarning,
                stacklevel=2,
            )
        candidates = all_neighbors_within_cutoff(
            xyz[center_indices], xyz[neighbor_indices], cell=cell,
            periodic=frame_periodic, cutoff=request.neighbor_cutoff,
        )
        near_proton = _has_neighbor_within_cutoff(
            xyz[center_indices], xyz[proton_indices], cell=cell,
            periodic=frame_periodic, cutoff=float(request.proton_cutoff),
        )
        iteration = _frame_iteration(trajectory, frame)
        output_frames.append(output_frame)
        output_iterations.append(iteration)

        for center_row, center_index in enumerate(center_indices):
            center_candidates = candidates[center_row]
            projections = np.asarray([value[2] @ c_hat for value in center_candidates], dtype=float)
            basal_positions = set(
                sorted(
                    range(len(center_candidates)),
                    key=lambda position: (
                        abs(float(projections[position])), center_candidates[position][1]
                    ),
                )[:3]
            )
            has_three = len(basal_positions) == 3
            center_label = str(labels[center_index])
            center_charge = (
                float(charge_array[frame, center_index])
                if charge_array is not None
                else float(formal.get(center_label.casefold(), np.nan))
            )
            center_rows.append({
                "frame_index": output_frame,
                "iter": iteration,
                "site_atom_index": int(center_index),
                "site_atom_id": int(ids[center_index]),
                "site_element": center_label,
                "site_x (angstrom)": float(xyz[center_index, 0]),
                "site_y (angstrom)": float(xyz[center_index, 1]),
                "site_z (angstrom)": float(xyz[center_index, 2]),
                "site_charge (e)": center_charge,
                "neighbor_count_within_cutoff": len(center_candidates),
                "has_three_basal_neighbors": has_three,
                "has_apical_neighbor": False,
                "has_proton_within_cutoff": bool(near_proton[center_row]),
                "proton_cutoff (angstrom)": float(request.proton_cutoff),
                "charge_source": "per-atom" if charge_array is not None else "formal",
                "neighbor_cutoff (angstrom)": (
                    np.nan if request.neighbor_cutoff is None else float(request.neighbor_cutoff)
                ),
                "c_axis_x": float(c_hat[0]), "c_axis_y": float(c_hat[1]),
                "c_axis_z": float(c_hat[2]),
                "periodic_a": periodic[0], "periodic_b": periodic[1], "periodic_c": periodic[2],
            })
            for position, (local_index, distance, vector) in enumerate(center_candidates):
                atom_index = int(neighbor_indices[local_index])
                label = str(labels[atom_index])
                image_xyz = xyz[center_index] + vector
                neighbor_rows.append({
                    "frame_index": output_frame, "iter": iteration,
                    "site_atom_id": int(ids[center_index]), "site_element": center_label,
                    "neighbor_rank": position + 1, "neighbor_atom_index": atom_index,
                    "neighbor_atom_id": int(ids[atom_index]), "neighbor_element": label,
                    "neighbor_charge (e)": (
                        float(charge_array[frame, atom_index])
                        if charge_array is not None
                        else float(formal.get(label.casefold(), np.nan))
                    ),
                    "neighbor_x (angstrom)": float(xyz[atom_index, 0]),
                    "neighbor_y (angstrom)": float(xyz[atom_index, 1]),
                    "neighbor_z (angstrom)": float(xyz[atom_index, 2]),
                    "neighbor_image_x (angstrom)": float(image_xyz[0]),
                    "neighbor_image_y (angstrom)": float(image_xyz[1]),
                    "neighbor_image_z (angstrom)": float(image_xyz[2]),
                    "distance (angstrom)": distance,
                    "bond_x (angstrom)": float(vector[0]),
                    "bond_y (angstrom)": float(vector[1]),
                    "bond_z (angstrom)": float(vector[2]),
                    "bond_c (angstrom)": float(projections[position]),
                    "neighbor_role": "basal" if position in basal_positions else "apical_candidate",
                })

    centers = pd.DataFrame(center_rows, columns=CENTER_COLUMNS)
    neighbors = pd.DataFrame(neighbor_rows, columns=NEIGHBOR_COLUMNS)
    return WurtziteNeighborResult(
        table=neighbors, request=request, centers=centers, neighbors=neighbors,
        frame_indices=np.asarray(output_frames, dtype=int),
        iterations=np.asarray(output_iterations, dtype=int),
    )


@register_task("get-three-folded-wurtzite-neighbors", label="Three-folded Wurtzite Neighbors")
class WurtziteNeighborTask(AnalysisTask):
    """Extract three basal neighbors and polarity-dependent apical candidates."""

    execution_capabilities = TaskCapabilities(
        shape=ExecutionShape.INDEPENDENT_FRAME_MAP, thread_safe=True, automatic_parallel=False,
        supports_selective_frames=True, estimated_frame_bytes=16 * 1024 * 1024,
    )
    required_data = TrajectoryData
    supports_selective_streaming = False
    VERSION = "1"

    def required_data_for(self, request: WurtziteNeighborRequest, args: dict | None = None):
        return required_wurtzite_data_type(request, args)

    @staticmethod
    def required_data_fields_for(request: WurtziteNeighborRequest, _args: dict) -> tuple[str, ...]:
        return ("trajectory", "charges") if request.charge_source != "formal" else ("trajectory",)

    @staticmethod
    def recommended_presentations(
        _result: WurtziteNeighborResult, _payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        return [PresentationSpec(renderer="table", label="Three-folded neighbors", view_type="table")]

    def run(self, data, request: WurtziteNeighborRequest, reporter=None):
        _ = reporter
        trajectory, charges = _charges_for_request(data, request)
        return extract_wurtzite_neighbors(trajectory, request, charges=charges)

    def run_stream(self, frames, request, reporter=None, pipeline=None):
        return stream_neighbors(self, frames, request, WurtziteNeighborResult,
                                CENTER_COLUMNS, NEIGHBOR_COLUMNS, pipeline=pipeline, reporter=reporter)


__all__ = [
    "CENTER_COLUMNS", "NEIGHBOR_COLUMNS", "WurtziteNeighborRequest",
    "WurtziteNeighborResult", "WurtziteNeighborTask", "all_neighbors_within_cutoff",
    "extract_wurtzite_neighbors",
]
