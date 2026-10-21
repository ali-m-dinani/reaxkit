"""Al-centered hBN dipoles with conservative or locally neutral charge weights."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field as dc_field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from reaxkit.core.runtime.execution_contracts import TaskCapabilities, ExecutionShape
from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.neighbors import (
    _frame_labels,
    _source_frame,
    _trajectory_and_charges,
    required_wurtzite_data_type,
)
from reaxkit.analysis.ferroelectrics.hbn_reference.polarization import (
    ELECTRON_CHARGE_SIGN,
    HBNReferencePolarizationRequest,
    HBNReferencePolarizationResult,
    _frame_cell,
    _unit_vector,
    _bbox_volume,
    _hull_volume,
    calculate_hbn_reference_polarization,
)
from reaxkit.analysis.ferroelectrics.hbn_reference.al_neighborhoods import (
    build_al_neighborhoods,
    neighborhood_hull_volumes,
)
from reaxkit.analysis.ferroelectrics.poled_counts import directional_poled_counts
from reaxkit.core.platform.constants import const
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ElectrostaticsData, TrajectoryData
from reaxkit.engine.common.generators.extended_xyz_generator import (
    ExtendedXYZFrame,
    ExtendedXYZWriter,
)
from reaxkit.presentation.specs import PresentationSpec

LocalVolumeMethod = Literal["equal", "deformation", "hull"]
LocalChargeTreatment = Literal["auto", "raw", "neutralize"]
LocalGrouping = Literal["al", "cell", "layer"]


@dataclass
class HBNReferenceLocalPolarizationRequest(HBNReferencePolarizationRequest):
    """Configure Al-centered charge weights, geometric signs, and local volumes."""

    include_displacements: bool = True
    volume_method: Literal["bbox", "hull", "cell", "local-hull"] = "bbox"
    local_volume_method: LocalVolumeMethod = dc_field(
        default="equal",
        metadata={
            "label": "Local volume method",
            "choices": ["equal", "deformation", "hull"],
        },
    )
    deformation_neighbors: int = 12
    local_charge_method: Literal["partition", "neutral-center"] = dc_field(
        default="partition",
        metadata={"label": "Local charge method", "choices": ["partition", "neutral-center"]},
    )
    geometric_zero_tolerance: float = 1.0e-6
    local_charge_treatment: LocalChargeTreatment | None = dc_field(
        default=None,
        metadata={
            "label": "Local charge treatment",
            "choices": ["auto", "raw", "neutralize"],
        },
    )
    local_grouping: LocalGrouping = dc_field(
        default="al",
        metadata={
            "label": "Local grouping",
            "choices": ["al", "cell", "layer"],
        },
    )


@dataclass
class HBNReferenceLocalPolarizationResult(BaseResult):
    """Cell- and layer-resolved dipoles, assigned volumes, and polarizations."""

    table: pd.DataFrame
    summary: pd.DataFrame
    cell_table: pd.DataFrame
    cell_summary: pd.DataFrame
    layer_table: pd.DataFrame
    layer_summary: pd.DataFrame
    poled_counts: pd.DataFrame
    request: HBNReferenceLocalPolarizationRequest
    reference_result: HBNReferencePolarizationResult
    trajectory: TrajectoryData
    frame_indices: np.ndarray
    iterations: np.ndarray

    @property
    def csv_tables(self) -> dict[str, pd.DataFrame]:
        return {
            "hbn_reference_local_polarization": self.table,
            "hbn_reference_local_polarization_summary": self.summary,
            "hbn_reference_cell_polarization": self.cell_table,
            "hbn_reference_cell_polarization_summary": self.cell_summary,
            "hbn_reference_layer_polarization": self.layer_table,
            "hbn_reference_layer_polarization_summary": self.layer_summary,
            "hbn_reference_local_polarization_poled_counts": self.poled_counts,
        }


def _validate_request(request: HBNReferenceLocalPolarizationRequest) -> None:
    if request.local_volume_method not in {"equal", "deformation", "hull"}:
        raise ValueError("local_volume_method must be 'equal', 'deformation', or 'hull'.")
    if request.volume_method == "local-hull" and request.local_volume_method == "deformation":
        raise ValueError("volume_method='local-hull' cannot be combined with local_volume_method='deformation'.")
    if int(request.deformation_neighbors) < 3:
        raise ValueError("deformation_neighbors must be at least 3.")
    if request.local_charge_treatment is not None:
        raise ValueError("Use local_charge_method='partition' or 'neutral-center'; legacy local_charge_treatment is no longer supported.")
    if request.local_charge_method not in {"partition", "neutral-center"}:
        raise ValueError("local_charge_method must be 'partition' or 'neutral-center'.")
    if not np.isfinite(request.geometric_zero_tolerance) or request.geometric_zero_tolerance < 0:
        raise ValueError("geometric_zero_tolerance must be finite and non-negative.")
    if request.local_grouping not in {"al", "cell", "layer"}:
        raise ValueError("local_grouping must be 'al', 'cell', or 'layer'.")


def _al_group_rows(reference_result, request, trajectory) -> pd.DataFrame:
    """Partition atomic contributions or use neutral center-based neighborhoods."""
    neighborhoods = build_al_neighborhoods(reference_result.reference)
    al_indices = neighborhoods.al_indices
    nitrogen_indices = neighborhoods.nitrogen_indices
    mapping = reference_result.mapping.sort_values("atom_index")
    representatives = mapping.iloc[al_indices]
    rows = []
    for (frame, iteration), atomic in reference_result.displacements.groupby(["frame_index", "iter"], sort=True):
        atomic = atomic.sort_values("atom_index")
        displacement = atomic[[f"displacement_{axis} (angstrom)" for axis in "xyzc"]].to_numpy(float)
        charges = atomic["charge (e)"].to_numpy(float)
        al_charges = charges[al_indices]
        nitrogen_charges = charges[nitrogen_indices] * neighborhoods.nitrogen_weights
        partition = ELECTRON_CHARGE_SIGN * (
            al_charges[:, None] * displacement[al_indices]
            + (nitrogen_charges[:, :, None] * displacement[nitrogen_indices]).sum(axis=1)
        )
        cell = _frame_cell(trajectory, request, int(frame))
        relative_xyz = neighborhoods.relative_displacement(displacement[:, :3], cell, request.periodic)
        relative = np.column_stack((relative_xyz, relative_xyz @ _unit_vector(request.c_axis, "c_axis")))
        neutral = ELECTRON_CHARGE_SIGN * al_charges[:, None] * relative
        net_charge = al_charges + nitrogen_charges.sum(axis=1)
        selected = partition if request.local_charge_method == "partition" else neutral
        table = pd.DataFrame({
            "frame_index": int(frame), "iter": int(iteration),
            "local_neighborhood_id": neighborhoods.layer_ids,
            "local_cell_id": neighborhoods.cell_ids,
            "local_grouping": "al",
            "atom_count": 4, "neighbor_count": 3,
            "effective_atom_count": 1.0 + neighborhoods.nitrogen_weights.sum(axis=1),
            "net_charge (e)": net_charge,
            "effective_net_charge (e)": net_charge if request.local_charge_method == "partition" else 0.0,
            "al_charge (e)": al_charges,
            "local_charge_method": request.local_charge_method,
            "charge_source": atomic["charge_source"].iloc[0],
            "charge_neutralization_applied": request.local_charge_method == "neutral-center",
            "representative_atom_index": al_indices,
            "representative_atom_id": representatives["atom_id"].to_numpy(int),
            "representative_element": representatives["element"].to_numpy(str),
            "composition": "Al1,N(shared)",
        })
        if request.local_volume_method == "hull" or request.volume_method == "local-hull":
            vectors = neighborhoods.current_vectors(
                displacement[:, :3], reference_result.reference.atoms.cell.array, cell, request.periodic,
            )
            table["neighborhood_hull_volume (angstrom^3)"] = neighborhood_hull_volumes(vectors)
        for component, axis in enumerate("xyzc"):
            table[f"raw_dipole_{axis} (e*angstrom)"] = partition[:, component]
            table[f"neutralized_dipole_{axis} (e*angstrom)"] = neutral[:, component]
            table[f"dipole_{axis} (e*angstrom)"] = selected[:, component]
            table[f"geometric_displacement_{axis} (angstrom)"] = relative[:, component]
            table[f"geometric_polarity_{axis}"] = np.where(
                np.abs(relative[:, component]) <= request.geometric_zero_tolerance, 0.0,
                np.sign(relative[:, component]),
            )
            if axis in "xyz":
                table[f"reference_center_{axis} (angstrom)"] = neighborhoods.reference_centers[:, component]
                table[f"center_{axis} (angstrom)"] = neighborhoods.reference_centers[:, component] + displacement[al_indices, component]
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def _aggregate_al_cells(table) -> pd.DataFrame:
    """Sum Al contributions and assigned volumes into legacy reference cells."""
    aggregations = {}
    summed = ("dipole_", "raw_dipole_", "neutralized_dipole_", "P_")
    for column in table.columns:
        if column in {"frame_index", "iter", "local_cell_id", "local_neighborhood_id"}:
            continue
        if column.startswith(summed) or column in {
            "atom_count", "effective_atom_count", "neighbor_count", "net_charge (e)",
            "effective_net_charge (e)", "local_volume (angstrom^3)", "local_volume_weight",
            "neighborhood_hull_volume (angstrom^3)",
        }:
            aggregations[column] = "sum"
        elif column.startswith(("center_", "reference_center_", "geometric_displacement_")):
            aggregations[column] = "mean"
        elif column.startswith("geometric_polarity_"):
            continue
        elif column.startswith("has_valid_"):
            aggregations[column] = "all"
        else:
            aggregations[column] = "first"
    result = table.groupby(["frame_index", "iter", "local_cell_id"], sort=True).agg(aggregations).reset_index()
    result["local_grouping"] = "cell"
    result["composition"] = "Al2,N(shared)"
    factor = float(const("ea3_to_uC_cm2"))
    volumes = result["local_volume (angstrom^3)"].to_numpy(float)
    valid = np.isfinite(volumes) & (volumes > 0.0)
    result["has_valid_local_volume"] = valid
    result["local_volume_status"] = np.where(valid, "ok", "invalid")
    for axis in "xyzc":
        result[f"P_{axis} (uC/cm^2)"] = np.divide(
            result[f"dipole_{axis} (e*angstrom)"].to_numpy(float) * factor,
            volumes, out=np.full(len(result), np.nan), where=valid,
        )
    return result


def _periodic_shifts(cell: np.ndarray, periodic: tuple[bool, bool, bool]) -> np.ndarray:
    ranges = [(-1, 0, 1) if enabled else (0,) for enabled in periodic]
    fractional = np.asarray(list(itertools.product(*ranges)), dtype=float)
    return fractional @ np.asarray(cell, dtype=float)


def _deformation_weights(
        reference_centers: np.ndarray,
        current_centers: np.ndarray,
        cell: np.ndarray,
        periodic: tuple[bool, bool, bool],
        neighbor_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate relative cell volumes from a local affine deformation fit.

    The determinant of the best-fit local deformation gradient is used only as
    a relative weight. The caller normalizes all weights to the selected frame
    volume, so local volumes add exactly to that global volume.
    """

    count = len(reference_centers)
    weights = np.full(count, np.nan)
    valid = np.zeros(count, dtype=bool)
    if count < 4:
        return np.ones(count), valid

    shifts = _periodic_shifts(cell, periodic)
    image_positions = (
            reference_centers[None, :, :] + shifts[:, None, :]
    ).reshape(-1, 3)
    image_source = np.tile(np.arange(count, dtype=int), len(shifts))
    image_shift = np.repeat(shifts, count, axis=0)
    tree = cKDTree(image_positions)
    query_count = min(len(image_positions), max(16, 4 * (int(neighbor_count) + 1)))

    for center_id in range(count):
        _, image_indices = tree.query(reference_centers[center_id], k=query_count)
        selected: list[int] = []
        seen: set[int] = set()
        for image_index in np.atleast_1d(image_indices).astype(int):
            source_id = int(image_source[image_index])
            if source_id == center_id or source_id in seen:
                continue
            seen.add(source_id)
            selected.append(image_index)
            if len(selected) >= min(int(neighbor_count), count - 1):
                break
        if len(selected) < 3:
            continue
        selected_array = np.asarray(selected, dtype=int)
        sources = image_source[selected_array]
        neighbor_shifts = image_shift[selected_array]
        reference_vectors = (
                reference_centers[sources] + neighbor_shifts - reference_centers[center_id]
        )
        current_vectors = (
                current_centers[sources] + neighbor_shifts - current_centers[center_id]
        )
        if np.linalg.matrix_rank(reference_vectors) < 3:
            continue
        deformation, *_ = np.linalg.lstsq(
            reference_vectors, current_vectors, rcond=None
        )
        determinant = abs(float(np.linalg.det(deformation)))
        if np.isfinite(determinant) and determinant > 0.0:
            weights[center_id] = determinant
            valid[center_id] = True

    replacement = float(np.median(weights[valid])) if np.any(valid) else 1.0
    weights[~valid] = replacement
    return weights, valid


def _assign_local_volumes(
        table: pd.DataFrame,
        reference_result: HBNReferencePolarizationResult,
        trajectory: TrajectoryData,
        request: HBNReferenceLocalPolarizationRequest,
        factor: float,
) -> pd.DataFrame:
    """Assign shared slab volumes or independent neighborhood hull volumes."""

    table = table.copy()
    local_method = "hull" if request.volume_method == "local-hull" else request.local_volume_method
    table["local_volume_method"] = local_method
    table["frame_volume_method"] = request.volume_method
    table["frame_volume (angstrom^3)"] = np.nan
    table["local_volume_weight"] = np.nan
    table["has_valid_local_deformation"] = False
    table["local_volume (angstrom^3)"] = np.nan

    global_by_frame = reference_result.table.set_index("frame_index")
    periodic = tuple(bool(value) for value in request.periodic)
    for frame, indices in table.groupby("frame_index", sort=False).groups.items():
        frame = int(frame)
        frame_indices = np.asarray(list(indices), dtype=int)
        frame_volume = float(global_by_frame.loc[frame, "volume (angstrom^3)"])
        if request.volume_method in {"bbox", "hull", "local-hull"}:
            atomic = reference_result.displacements.loc[reference_result.displacements["frame_index"].eq(frame)]
            unwrapped = (
                atomic[[f"reference_{axis} (angstrom)" for axis in "xyz"]].to_numpy(float)
                + atomic[[f"displacement_{axis} (angstrom)" for axis in "xyz"]].to_numpy(float)
            )
            estimator = _hull_volume if request.volume_method == "hull" else _bbox_volume
            frame_volume = float(estimator(unwrapped))
        count = len(frame_indices)
        if local_method == "hull":
            volumes = table.loc[frame_indices, "neighborhood_hull_volume (angstrom^3)"].to_numpy(float)
            table.loc[frame_indices, "frame_volume (angstrom^3)"] = frame_volume
            table.loc[frame_indices, "local_volume (angstrom^3)"] = volumes
            continue
        if local_method == "equal":
            weights = np.ones(count, dtype=float)
            deformation_valid = np.ones(count, dtype=bool)
        else:
            reference_centers = table.loc[
                frame_indices,
                [
                    "reference_center_x (angstrom)",
                    "reference_center_y (angstrom)",
                    "reference_center_z (angstrom)",
                ],
            ].to_numpy(float)
            current_centers = table.loc[
                frame_indices,
                [
                    "center_x (angstrom)",
                    "center_y (angstrom)",
                    "center_z (angstrom)",
                ],
            ].to_numpy(float)
            cell = _frame_cell(trajectory, request, frame)
            weights, deformation_valid = _deformation_weights(
                reference_centers,
                current_centers,
                cell,
                periodic,
                int(request.deformation_neighbors),
            )
        weight_sum = float(np.sum(weights))
        volumes = (
            weights / weight_sum * frame_volume
            if np.isfinite(frame_volume) and frame_volume > 0.0 and weight_sum > 0.0
            else np.full(count, np.nan)
        )
        table.loc[frame_indices, "frame_volume (angstrom^3)"] = frame_volume
        table.loc[frame_indices, "local_volume_weight"] = weights
        table.loc[frame_indices, "has_valid_local_deformation"] = deformation_valid
        table.loc[frame_indices, "local_volume (angstrom^3)"] = volumes

    local_volume = table["local_volume (angstrom^3)"].to_numpy(float)
    valid_volume = np.isfinite(local_volume) & (local_volume > 0.0)
    table["has_valid_local_volume"] = valid_volume
    table["local_volume_status"] = np.where(
        valid_volume, "ok", "degenerate_hull" if local_method == "hull" else "invalid",
    )
    for axis in (*"xyz", "c"):
        dipole = table[f"dipole_{axis} (e*angstrom)"].to_numpy(float)
        table[f"P_{axis} (uC/cm^2)"] = np.divide(
            dipole * factor,
            local_volume,
            out=np.full(len(table), np.nan),
            where=valid_volume & np.isfinite(dipole),
        )
    return table


def _summarize_local_table(
        table: pd.DataFrame,
        reference_result: HBNReferencePolarizationResult,
        request: HBNReferenceLocalPolarizationRequest,
        factor: float,
        *,
        grouping: LocalGrouping,
) -> pd.DataFrame:
    """Summarize volume and dipole closure for one local grouping."""

    global_by_frame = reference_result.table.set_index("frame_index")
    summary_rows: list[dict[str, object]] = []
    for (frame, iteration), group in table.groupby(["frame_index", "iter"], sort=True):
        valid = group["has_valid_local_volume"].to_numpy(bool)
        assigned_volume = float(group.loc[valid, "local_volume (angstrom^3)"].sum())
        global_row = global_by_frame.loc[int(frame)]
        row: dict[str, object] = {
            "frame_index": int(frame),
            "iter": int(iteration),
            "local_grouping": grouping,
            "local_group_count": len(group),
            f"local_{grouping}_count": len(group),
            "valid_local_volume_count": int(np.count_nonzero(valid)),
            "valid_local_deformation_count": int(
                group["has_valid_local_deformation"].astype(bool).sum()
            ),
            "local_volume_method": "hull" if request.volume_method == "local-hull" else request.local_volume_method,
            "local_charge_method": request.local_charge_method,
            "frame_volume_method": request.volume_method,
            "frame_volume (angstrom^3)": float(group["frame_volume (angstrom^3)"].iloc[0]),
            "assigned_volume (angstrom^3)": assigned_volume,
            "volume_partition_closure_error (angstrom^3)": assigned_volume - float(group["frame_volume (angstrom^3)"].iloc[0]),
            "invalid_local_volume_count": int(np.count_nonzero(~valid)),
        }
        for axis in (*"xyz", "c"):
            local_dipole = float(group[f"dipole_{axis} (e*angstrom)"].sum())
            raw_local_dipole = float(
                group[f"raw_dipole_{axis} (e*angstrom)"].sum()
            )
            global_dipole = float(global_row[f"dipole_{axis} (e*angstrom)"])
            row[f"dipole_{axis} (e*angstrom)"] = local_dipole
            row[f"raw_dipole_{axis} (e*angstrom)"] = raw_local_dipole
            row[f"global_dipole_{axis} (e*angstrom)"] = global_dipole
            row[f"dipole_closure_error_{axis} (e*angstrom)"] = (
                    local_dipole - global_dipole
            )
            row[f"raw_dipole_closure_error_{axis} (e*angstrom)"] = (
                    raw_local_dipole - global_dipole
            )
            row[f"P_{axis} (uC/cm^2)"] = (
                local_dipole / assigned_volume * factor
                if assigned_volume > 0.0
                else np.nan
            )
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def calculate_hbn_reference_local_polarization(
        data: TrajectoryData | ElectrostaticsData,
        request: HBNReferenceLocalPolarizationRequest,
    prepared=None,
) -> HBNReferenceLocalPolarizationResult:
    """Calculate cell- and layer-resolved dipoles and local polarization."""

    _validate_request(request)
    trajectory, _ = _trajectory_and_charges(data)
    global_request = replace(request, volume_method="bbox") if request.volume_method == "local-hull" else request
    reference_result = calculate_hbn_reference_polarization(data, global_request, prepared=prepared)
    factor_value = const("ea3_to_uC_cm2")
    if factor_value is None:  # pragma: no cover
        raise RuntimeError("The dipole-to-polarization conversion constant is missing.")
    factor = float(factor_value)

    layer_table = _assign_local_volumes(
        _al_group_rows(reference_result, request, trajectory),
        reference_result,
        trajectory,
        request,
        factor,
    )
    cell_table = _aggregate_al_cells(layer_table)
    layer_table = layer_table.drop(columns=["local_cell_id"])
    cell_summary = _summarize_local_table(
        cell_table, reference_result, request, factor, grouping="cell"
    )
    layer_summary = _summarize_local_table(
        layer_table, reference_result, request, factor, grouping="layer"
    )
    table = cell_table if request.local_grouping == "cell" else layer_table
    summary = cell_summary if request.local_grouping == "cell" else layer_summary
    table = table.copy()
    summary = summary.copy()
    table["local_grouping"] = request.local_grouping
    summary["local_grouping"] = request.local_grouping
    if request.local_grouping == "al":
        summary = summary.rename(columns={"local_layer_count": "local_al_count"})

    poled_counts = directional_poled_counts(
        table,
        trajectory,
        {axis: f"P_{axis} (uC/cm^2)" for axis in "xyz"},
    )
    return HBNReferenceLocalPolarizationResult(
        table=table,
        summary=summary,
        cell_table=cell_table,
        cell_summary=cell_summary,
        layer_table=layer_table,
        layer_summary=layer_summary,
        poled_counts=poled_counts,
        request=request,
        reference_result=reference_result,
        trajectory=trajectory,
        frame_indices=reference_result.frame_indices,
        iterations=reference_result.iterations,
    )


@register_task(
    "get-hbn-reference-local-polarization",
    label="h-BN-reference Local Polarization",
)
class HBNReferenceLocalPolarizationTask(AnalysisTask):
    """Calculate dipole and polarization for reference cells and layers."""

    required_data = TrajectoryData
    supports_output_profiles = True
    execution_capabilities = TaskCapabilities(shape=ExecutionShape.REFERENCE_FRAME_MAP,
        thread_safe=True, automatic_parallel=False, supports_selective_frames=True,
        reference_fields=("reference_frame",), estimated_frame_bytes=16 * 1024 * 1024)
    supports_selective_streaming = False
    VERSION = "6"

    def required_data_for(
            self, request: HBNReferenceLocalPolarizationRequest, args: dict | None = None
    ):
        return required_wurtzite_data_type(request, args)

    @staticmethod
    def required_data_fields_for(
            request: HBNReferenceLocalPolarizationRequest, args: dict
    ) -> tuple[str, ...]:
        return (
            ("trajectory",)
            if required_wurtzite_data_type(request, args) is TrajectoryData
            else ("trajectory", "charges")
        )

    @staticmethod
    def recommended_presentations(
            _result: HBNReferenceLocalPolarizationResult, _payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        return [
            PresentationSpec(
                renderer="table",
                label="h-BN-reference local polarization",
                view_type="table",
            )
        ]

    def run(self, data, request: HBNReferenceLocalPolarizationRequest, reporter=None):
        _ = reporter
        return calculate_hbn_reference_local_polarization(data, request)

    def run_stream(self, frames, request, reporter=None, pipeline=None):
        from reaxkit.analysis.ferroelectrics.polarization_stream import stream_polarization
        return stream_polarization(self, frames, request, "hbn_local", reporter, pipeline)


def _table_for_trajectory_frame(
        result: HBNReferenceLocalPolarizationResult,
        frame: int,
        table: pd.DataFrame | None = None,
) -> tuple[int, pd.DataFrame]:
    """Return source-frame rows for one compact in-memory trajectory frame."""

    source_frame = _source_frame(result.trajectory, int(frame)) if result.trajectory is not None else int(frame)
    source_table = result.table if table is None else table
    from reaxkit.core.runtime.result_store import ResultTable
    if isinstance(source_table, ResultTable):
        return source_frame, source_table.select_frame(source_frame)
    frame_numbers = source_table["frame_index"].astype(int)
    frame_table = source_table[frame_numbers.eq(source_frame)]
    if frame_table.empty and source_frame != int(frame):
        frame_table = source_table[frame_numbers.eq(int(frame))]
    return source_frame, frame_table


def write_local_polarization_extxyz(
        result: HBNReferenceLocalPolarizationResult,
        path: str | Path,
        *,
        precision: int = 8,
) -> Path:
    """Write one set of neighborhood properties on Al centers, never shared N."""

    trajectory = result.trajectory
    atom_ids = np.asarray(trajectory.atom_ids, dtype=int)
    id_to_index = {int(atom_id): index for index, atom_id in enumerate(atom_ids)}
    destination = Path(path)
    with ExtendedXYZWriter(destination, precision=precision) as writer:
        for frame in np.asarray(result.frame_indices, dtype=int):
            positions = np.asarray(trajectory.positions[frame], dtype=float)
            labels = _frame_labels(trajectory, int(frame)).astype(str)
            source_frame, neighborhoods = _table_for_trajectory_frame(
                result, int(frame), result.layer_table
            )
            finite = np.isfinite(positions).all(axis=1)
            representative_indices = np.fromiter(
                (id_to_index.get(int(atom_id), -1) for atom_id in neighborhoods["representative_atom_id"]),
                dtype=int, count=len(neighborhoods),
            )
            represented = representative_indices >= 0
            target = representative_indices[represented]
            rows = neighborhoods.loc[represented]
            neighborhood_ids = np.full(len(atom_ids), -1, dtype=int)
            neighborhood_ids[target] = rows["local_neighborhood_id"].to_numpy(int)
            is_center = np.zeros(len(atom_ids), dtype=int)
            is_center[target] = 1
            valid_volume = np.zeros(len(atom_ids), dtype=int)
            valid_volume[target] = rows["has_valid_local_volume"].to_numpy(bool).astype(int)
            properties = {
                "atom_number": atom_ids,
                "local_neighborhood_id": neighborhood_ids,
                "is_local_center": is_center,
                "has_valid_local_volume": valid_volume,
            }
            scalar_columns = {
                "local_volume": "local_volume (angstrom^3)",
                "local_dipole_c": "dipole_c (e*angstrom)",
                "local_polarization_c": "P_c (uC/cm^2)",
                "geometric_displacement_c": "geometric_displacement_c (angstrom)",
                "geometric_polarity_c": "geometric_polarity_c",
                "neighborhood_net_charge": "effective_net_charge (e)",
            }
            for name, column in scalar_columns.items():
                values = np.full(len(atom_ids), np.nan)
                values[target] = rows[column].to_numpy(float)
                properties[name] = values
            vector_columns = {
                "local_dipole": "dipole_{} (e*angstrom)",
                "local_polarization": "P_{} (uC/cm^2)",
                "geometric_displacement": "geometric_displacement_{} (angstrom)",
                "geometric_polarity": "geometric_polarity_{}",
            }
            for name, template in vector_columns.items():
                values = np.full((len(atom_ids), 3), np.nan)
                values[target] = rows[[template.format(axis) for axis in "xyz"]].to_numpy(float)
                properties[name] = values
            try:
                lattice = _frame_cell(trajectory, result.request, int(frame))
            except ValueError:
                lattice = None
            iteration = int(neighborhoods["iter"].iloc[0]) if not neighborhoods.empty else int(frame)
            writer.write_frame(
                ExtendedXYZFrame(
                    species=labels[finite].tolist(),
                    positions=positions[finite],
                    properties={name: values[finite] for name, values in properties.items()},
                    frame=source_frame,
                    iteration=iteration,
                    lattice=lattice,
                    pbc=tuple(bool(value) for value in result.request.periodic) if lattice is not None else None,
                    metadata={
                        "local_volume_method": "hull" if result.request.volume_method == "local-hull" else result.request.local_volume_method,
                        "local_grouping": "neighborhood",
                        "frame_volume_method": result.request.volume_method,
                        "local_charge_method": result.request.local_charge_method,
                        "geometric_zero_tolerance": result.request.geometric_zero_tolerance,
                        "geometric_polarity_convention": "Al_minus_mean_N_displacement",
                        "dipole_units": "e*angstrom",
                        "polarization_units": "uC/cm^2",
                        "local_volume_units": "angstrom^3",
                        "geometric_displacement_units": "angstrom",
                        "neighborhood_net_charge_units": "e",
                    },
                )
            )
    return destination


__all__ = [
    "HBNReferenceLocalPolarizationRequest",
    "HBNReferenceLocalPolarizationResult",
    "HBNReferenceLocalPolarizationTask",
    "LocalChargeTreatment",
    "LocalGrouping",
    "LocalVolumeMethod",
    "calculate_hbn_reference_local_polarization",
    "write_local_polarization_extxyz",
]
