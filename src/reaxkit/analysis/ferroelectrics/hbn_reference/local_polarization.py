"""Local dipole and polarization for neutral h-BN-reference primitive cells."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

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
    calculate_hbn_reference_polarization,
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

LocalVolumeMethod = Literal["equal", "deformation"]
LocalChargeTreatment = Literal["auto", "raw", "neutralize"]
LocalGrouping = Literal["cell", "layer"]


@dataclass
class HBNReferenceLocalPolarizationRequest(HBNReferencePolarizationRequest):
    """Configure neutral-cell dipoles and their local-volume convention."""

    include_displacements: bool = True
    local_volume_method: LocalVolumeMethod = dc_field(
        default="equal",
        metadata={
            "label": "Local volume method",
            "choices": ["equal", "deformation"],
        },
    )
    deformation_neighbors: int = 12
    local_charge_treatment: LocalChargeTreatment = dc_field(
        default="auto",
        metadata={
            "label": "Local charge treatment",
            "choices": ["auto", "raw", "neutralize"],
        },
    )
    local_grouping: LocalGrouping = dc_field(
        default="cell",
        metadata={
            "label": "Local grouping",
            "choices": ["cell", "layer"],
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
    if request.local_volume_method not in {"equal", "deformation"}:
        raise ValueError("local_volume_method must be 'equal' or 'deformation'.")
    if int(request.deformation_neighbors) < 3:
        raise ValueError("deformation_neighbors must be at least 3.")
    if request.local_charge_treatment not in {"auto", "raw", "neutralize"}:
        raise ValueError(
            "local_charge_treatment must be 'auto', 'raw', or 'neutralize'."
        )
    if request.local_grouping not in {"cell", "layer"}:
        raise ValueError("local_grouping must be 'cell' or 'layer'.")


def _group_rows(
        reference_result: HBNReferencePolarizationResult,
        request: HBNReferenceLocalPolarizationRequest,
        *,
        grouping: LocalGrouping,
) -> pd.DataFrame:
    """Sum atomic dipoles within replicated crystallographic cells or layers."""

    displacement = reference_result.displacements
    group_column = "local_cell_id" if grouping == "cell" else "local_layer_id"
    keys = ["frame_index", "iter", group_column]
    aggregations: dict[str, tuple[str, str]] = {
        "atom_count": ("atom_index", "size"),
        "net_charge (e)": ("charge (e)", "sum"),
        "charge_source": ("charge_source", "first"),
    }
    for axis in "xyz":
        aggregations[f"reference_center_{axis} (angstrom)"] = (
            f"reference_{axis} (angstrom)",
            "mean",
        )
        aggregations[f"mean_displacement_{axis}"] = (
            f"displacement_{axis} (angstrom)",
            "mean",
        )
        aggregations[f"sum_displacement_{axis}"] = (
            f"displacement_{axis} (angstrom)",
            "sum",
        )
        aggregations[f"raw_dipole_{axis} (e*angstrom)"] = (
            f"dipole_{axis} (e*angstrom)",
            "sum",
        )
    aggregations["sum_displacement_c"] = ("displacement_c (angstrom)", "sum")
    aggregations["raw_dipole_c (e*angstrom)"] = (
        "dipole_c (e*angstrom)",
        "sum",
    )
    table = (
        displacement.groupby(keys, sort=True, observed=True)
        .agg(**aggregations)
        .reset_index()
    )

    mapping = reference_result.mapping.copy()
    mapping["_is_n"] = mapping["reference_element"].astype(str).str.casefold().eq("n")
    representatives = (
        mapping.sort_values([group_column, "_is_n", "atom_index"])
        .groupby(group_column, sort=True, observed=True)
        .first()
        .reset_index()
        .rename(
            columns={
                "atom_index": "representative_atom_index",
                "atom_id": "representative_atom_id",
                "element": "representative_element",
            }
        )
    )
    compositions = (
        mapping.groupby([group_column, "reference_element"], observed=True)
        .size()
        .rename("count")
        .reset_index()
        .sort_values([group_column, "reference_element"])
        .groupby(group_column, sort=True, observed=True)
        .apply(
            lambda group: ",".join(
                f"{element}{count}"
                for element, count in zip(group["reference_element"], group["count"])
            ),
            include_groups=False,
        )
        .rename("composition")
        .reset_index()
    )
    table = table.merge(
        representatives[
            [
                group_column,
                "representative_atom_index",
                "representative_atom_id",
                "representative_element",
            ]
        ],
        on=group_column,
        how="left",
        validate="many_to_one",
    ).merge(
        compositions,
        on=group_column,
        how="left",
        validate="many_to_one",
    )

    net_charge = table["net_charge (e)"].to_numpy(float)
    atom_count = table["atom_count"].to_numpy(float)
    neutralize = np.full(len(table), request.local_charge_treatment == "neutralize")
    if request.local_charge_treatment == "auto":
        neutralize = np.abs(net_charge) > 1.0e-10
    table["effective_net_charge (e)"] = np.where(neutralize, 0.0, net_charge)
    table["requested_local_charge_treatment"] = request.local_charge_treatment
    table["local_charge_treatment"] = np.where(neutralize, "neutralize", "raw")
    table["charge_neutralization_applied"] = neutralize
    for axis in "xyz":
        reference_center = table[f"reference_center_{axis} (angstrom)"].to_numpy(float)
        table[f"center_{axis} (angstrom)"] = (
            reference_center + table.pop(f"mean_displacement_{axis}").to_numpy(float)
        )
        raw = table[f"raw_dipole_{axis} (e*angstrom)"].to_numpy(float)
        displacement_sum = table.pop(f"sum_displacement_{axis}").to_numpy(float)
        neutralized = (
            raw - ELECTRON_CHARGE_SIGN * net_charge / atom_count * displacement_sum
        )
        table[f"neutralized_dipole_{axis} (e*angstrom)"] = neutralized
        table[f"dipole_{axis} (e*angstrom)"] = np.where(neutralize, neutralized, raw)
    raw_c = table["raw_dipole_c (e*angstrom)"].to_numpy(float)
    neutralized_c = (
        raw_c
        - ELECTRON_CHARGE_SIGN
        * net_charge
        / atom_count
        * table.pop("sum_displacement_c").to_numpy(float)
    )
    table["neutralized_dipole_c (e*angstrom)"] = neutralized_c
    table["dipole_c (e*angstrom)"] = np.where(neutralize, neutralized_c, raw_c)
    table["local_grouping"] = grouping
    table["local_group_id"] = table[group_column].to_numpy(int)
    return table


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
    """Assign normalized local volumes and derive polarization components."""

    table = table.copy()
    table["local_volume_method"] = request.local_volume_method
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
        count = len(frame_indices)
        if request.local_volume_method == "equal":
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
            "local_volume_method": request.local_volume_method,
            "frame_volume_method": request.volume_method,
            "frame_volume (angstrom^3)": float(global_row["volume (angstrom^3)"]),
            "assigned_volume (angstrom^3)": assigned_volume,
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
) -> HBNReferenceLocalPolarizationResult:
    """Calculate cell- and layer-resolved dipoles and local polarization."""

    _validate_request(request)
    trajectory, _ = _trajectory_and_charges(data)
    reference_result = calculate_hbn_reference_polarization(data, request)
    factor_value = const("ea3_to_uC_cm2")
    if factor_value is None:  # pragma: no cover
        raise RuntimeError("The dipole-to-polarization conversion constant is missing.")
    factor = float(factor_value)

    cell_table = _assign_local_volumes(
        _group_rows(reference_result, request, grouping="cell"),
        reference_result,
        trajectory,
        request,
        factor,
    )
    layer_table = _assign_local_volumes(
        _group_rows(reference_result, request, grouping="layer"),
        reference_result,
        trajectory,
        request,
        factor,
    )
    cell_summary = _summarize_local_table(
        cell_table, reference_result, request, factor, grouping="cell"
    )
    layer_summary = _summarize_local_table(
        layer_table, reference_result, request, factor, grouping="layer"
    )
    table = cell_table if request.local_grouping == "cell" else layer_table
    summary = cell_summary if request.local_grouping == "cell" else layer_summary

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
    supports_selective_streaming = False
    VERSION = "3"

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


def _table_for_trajectory_frame(
        result: HBNReferenceLocalPolarizationResult,
        frame: int,
        table: pd.DataFrame | None = None,
) -> tuple[int, pd.DataFrame]:
    """Return source-frame rows for one compact in-memory trajectory frame."""

    source_frame = _source_frame(result.trajectory, int(frame))
    source_table = result.table if table is None else table
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
    """Write full frames with each local value on one representative atom."""

    trajectory = result.trajectory
    atom_ids = np.asarray(trajectory.atom_ids, dtype=int)
    id_to_index = {int(atom_id): index for index, atom_id in enumerate(atom_ids)}
    mapping = result.reference_result.mapping
    atom_local_cell_id = np.full(len(atom_ids), -1, dtype=int)
    atom_local_layer_id = np.full(len(atom_ids), -1, dtype=int)
    mapped_indices = np.fromiter(
        (id_to_index.get(int(atom_id), -1) for atom_id in mapping["atom_id"]),
        dtype=int,
        count=len(mapping),
    )
    mapped = mapped_indices >= 0
    atom_local_cell_id[mapped_indices[mapped]] = mapping.loc[
        mapped, "local_cell_id"
    ].to_numpy(int)
    atom_local_layer_id[mapped_indices[mapped]] = mapping.loc[
        mapped, "local_layer_id"
    ].to_numpy(int)
    destination = Path(path)
    with ExtendedXYZWriter(destination, precision=precision) as writer:
        for frame in np.asarray(result.frame_indices, dtype=int):
            positions = np.asarray(trajectory.positions[frame], dtype=float)
            labels = _frame_labels(trajectory, int(frame)).astype(str)
            source_frame, cell_frame_table = _table_for_trajectory_frame(
                result, int(frame), result.cell_table
            )
            _, layer_frame_table = _table_for_trajectory_frame(
                result, int(frame), result.layer_table
            )
            finite = np.isfinite(positions).all(axis=1)
            local_cell_id = atom_local_cell_id.copy()
            local_layer_id = atom_local_layer_id.copy()
            is_center = np.zeros(len(atom_ids), dtype=int)
            valid_volume = np.zeros(len(atom_ids), dtype=int)
            local_volume = np.full(len(atom_ids), np.nan)
            dipole = np.full((len(atom_ids), 3), np.nan)
            polarization = np.full((len(atom_ids), 3), np.nan)
            dipole_c = np.full(len(atom_ids), np.nan)
            polarization_c = np.full(len(atom_ids), np.nan)
            representative_indices = np.fromiter(
                (
                    id_to_index.get(int(atom_id), -1)
                    for atom_id in cell_frame_table["representative_atom_id"]
                ),
                dtype=int,
                count=len(cell_frame_table),
            )
            represented = representative_indices >= 0
            target = representative_indices[represented]
            represented_rows = cell_frame_table.loc[represented]
            is_center[target] = 1
            valid_volume[target] = represented_rows[
                "has_valid_local_volume"
            ].to_numpy(bool).astype(int)
            local_volume[target] = represented_rows[
                "local_volume (angstrom^3)"
            ].to_numpy(float)
            dipole[target] = represented_rows[
                [f"dipole_{axis} (e*angstrom)" for axis in "xyz"]
            ].to_numpy(float)
            polarization[target] = represented_rows[
                [f"P_{axis} (uC/cm^2)" for axis in "xyz"]
            ].to_numpy(float)
            dipole_c[target] = represented_rows["dipole_c (e*angstrom)"].to_numpy(float)
            polarization_c[target] = represented_rows["P_c (uC/cm^2)"].to_numpy(float)

            is_layer_center = np.zeros(len(atom_ids), dtype=int)
            valid_layer_volume = np.zeros(len(atom_ids), dtype=int)
            layer_volume = np.full(len(atom_ids), np.nan)
            layer_dipole = np.full((len(atom_ids), 3), np.nan)
            layer_polarization = np.full((len(atom_ids), 3), np.nan)
            layer_dipole_c = np.full(len(atom_ids), np.nan)
            layer_polarization_c = np.full(len(atom_ids), np.nan)
            layer_representative_indices = np.fromiter(
                (
                    id_to_index.get(int(atom_id), -1)
                    for atom_id in layer_frame_table["representative_atom_id"]
                ),
                dtype=int,
                count=len(layer_frame_table),
            )
            layer_represented = layer_representative_indices >= 0
            layer_target = layer_representative_indices[layer_represented]
            layer_rows = layer_frame_table.loc[layer_represented]
            is_layer_center[layer_target] = 1
            valid_layer_volume[layer_target] = layer_rows[
                "has_valid_local_volume"
            ].to_numpy(bool).astype(int)
            layer_volume[layer_target] = layer_rows[
                "local_volume (angstrom^3)"
            ].to_numpy(float)
            layer_dipole[layer_target] = layer_rows[
                [f"dipole_{axis} (e*angstrom)" for axis in "xyz"]
            ].to_numpy(float)
            layer_polarization[layer_target] = layer_rows[
                [f"P_{axis} (uC/cm^2)" for axis in "xyz"]
            ].to_numpy(float)
            layer_dipole_c[layer_target] = layer_rows[
                "dipole_c (e*angstrom)"
            ].to_numpy(float)
            layer_polarization_c[layer_target] = layer_rows[
                "P_c (uC/cm^2)"
            ].to_numpy(float)
            try:
                lattice = _frame_cell(trajectory, result.request, int(frame))
            except ValueError:
                lattice = None
            iteration = (
                int(cell_frame_table["iter"].iloc[0])
                if not cell_frame_table.empty
                else int(frame)
            )
            writer.write_frame(
                ExtendedXYZFrame(
                    species=labels[finite].tolist(),
                    positions=positions[finite],
                    properties={
                        "atom_number": atom_ids[finite],
                        "local_cell_id": local_cell_id[finite],
                        "local_layer_id": local_layer_id[finite],
                        "is_local_cell_center": is_center[finite],
                        "has_valid_local_volume": valid_volume[finite],
                        "local_volume": local_volume[finite],
                        "local_dipole": dipole[finite],
                        "local_polarization": polarization[finite],
                        "local_dipole_c": dipole_c[finite],
                        "local_polarization_c": polarization_c[finite],
                        "is_local_layer_center": is_layer_center[finite],
                        "has_valid_layer_volume": valid_layer_volume[finite],
                        "layer_volume": layer_volume[finite],
                        "layer_dipole": layer_dipole[finite],
                        "layer_polarization": layer_polarization[finite],
                        "layer_dipole_c": layer_dipole_c[finite],
                        "layer_polarization_c": layer_polarization_c[finite],
                    },
                    frame=source_frame,
                    iteration=iteration,
                    lattice=lattice,
                    pbc=(
                        tuple(bool(value) for value in result.request.periodic)
                        if lattice is not None
                        else None
                    ),
                    metadata={
                        "local_volume_method": result.request.local_volume_method,
                        "selected_local_grouping": result.request.local_grouping,
                        "frame_volume_method": result.request.volume_method,
                        "local_charge_treatment": result.request.local_charge_treatment,
                        "dipole_units": "e*angstrom",
                        "polarization_units": "uC/cm^2",
                        "local_volume_units": "angstrom^3",
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
