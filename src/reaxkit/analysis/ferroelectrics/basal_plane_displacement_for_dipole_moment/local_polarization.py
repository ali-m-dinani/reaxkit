"""Per-center polarization from basal-plane displacement dipoles."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis.electrostatics.electrostatics import _field_component_series
from reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.dipole import (
    BasalPlaneDipoleRequest,
    BasalPlaneDipoleResult,
    calculate_basal_plane_dipoles,
)
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.neighbors import (
    _frame_cell,
    _frame_labels,
    _source_frame,
    _trajectory_and_charges,
    required_wurtzite_data_type,
)
from reaxkit.analysis.ferroelectrics.poled_counts import directional_poled_counts
from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.polarization import (
    VolumeMethod,
    _bbox_volume,
    _cell_bin_volume,
    _hull_volume,
)
from reaxkit.core.platform.constants import const
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ElectricFieldData, ElectrostaticsData, TrajectoryData
from reaxkit.engine.common.generators.extended_xyz_generator import (
    ExtendedXYZFrame,
    ExtendedXYZWriter,
)
from reaxkit.presentation.specs import PresentationSpec

LocalVolumeMethod = Literal["equal", "coordination"]


@dataclass
class BasalPlaneLocalPolarizationRequest(BasalPlaneDipoleRequest):
    """Configure per-center polarization and its local-volume convention."""

    local_volume_method: LocalVolumeMethod = dc_field(
        default="equal",
        metadata={
            "label": "Local volume method",
            "choices": ["equal", "coordination"],
        },
    )
    volume_method: VolumeMethod = dc_field(
        default="hull",
        metadata={"label": "Frame volume method", "choices": ["hull", "bbox", "cell"]},
    )
    include_electric_field: bool = False
    field_direction: Literal["x", "y", "z"] = "z"


@dataclass
class BasalPlaneLocalPolarizationResult(BaseResult):
    """One local dipole, volume, and polarization vector per selected center."""

    table: pd.DataFrame
    request: BasalPlaneLocalPolarizationRequest
    summary: pd.DataFrame
    poled_counts: pd.DataFrame
    dipole_result: BasalPlaneDipoleResult
    trajectory: TrajectoryData
    electric_field: ElectricFieldData | None
    frame_indices: np.ndarray
    iterations: np.ndarray

    @property
    def csv_tables(self) -> dict[str, pd.DataFrame]:
        return {
            "basal_plane_local_polarization": self.table,
            "basal_plane_local_polarization_summary": self.summary,
            "basal_plane_local_polarization_poled_counts": self.poled_counts,
        }


def _validate_request(request: BasalPlaneLocalPolarizationRequest) -> None:
    if request.local_volume_method not in {"equal", "coordination"}:
        raise ValueError("local_volume_method must be 'equal' or 'coordination'.")
    if request.volume_method not in {"hull", "bbox", "cell"}:
        raise ValueError("volume_method must be 'hull', 'bbox', or 'cell'.")
    if request.field_direction not in {"x", "y", "z"}:
        raise ValueError("field_direction must be 'x', 'y', or 'z'.")


def _frame_volume(
    trajectory: TrajectoryData,
    request: BasalPlaneLocalPolarizationRequest,
    frame: int,
) -> float:
    if request.volume_method == "cell":
        return float(_cell_bin_volume(trajectory, request, frame, (1, 1, 1)))
    coordinates = np.asarray(trajectory.positions[frame], dtype=float)
    coordinates = coordinates[np.isfinite(coordinates).all(axis=1)]
    estimator = _hull_volume if request.volume_method == "hull" else _bbox_volume
    return float(estimator(coordinates))


def _coordination_volumes(dipoles: BasalPlaneDipoleResult) -> dict[tuple[int, int], float]:
    """Return volumes of N-neighbor tetrahedra around valid centers."""

    geometry = dipoles.polarity_result.neighbor_geometry
    volumes: dict[tuple[int, int], float] = {}
    for (frame, site_id), group in geometry.groupby(
        ["frame_index", "site_atom_id"], sort=False
    ):
        basal = group[group["neighbor_role"] == "basal"]
        apical = group[group["neighbor_role"] == "apical"]
        if len(basal) != 3 or len(apical) != 1:
            continue
        columns = [
            "neighbor_image_x (angstrom)",
            "neighbor_image_y (angstrom)",
            "neighbor_image_z (angstrom)",
        ]
        vertices = np.vstack(
            [basal[columns].to_numpy(float), apical[columns].to_numpy(float)]
        )
        if not np.isfinite(vertices).all():
            continue
        matrix = np.stack(
            (vertices[1] - vertices[0], vertices[2] - vertices[0], vertices[3] - vertices[0]),
            axis=0,
        )
        volume = abs(float(np.linalg.det(matrix))) / 6.0
        if np.isfinite(volume) and volume > 0.0:
            volumes[(int(frame), int(site_id))] = volume
    return volumes


def calculate_basal_plane_local_polarization(
    data,
    request: BasalPlaneLocalPolarizationRequest,
) -> BasalPlaneLocalPolarizationResult:
    """Normalize every valid center dipole by an assigned local volume.

    ``equal`` assigns each valid center an equal share of the selected frame
    volume. ``coordination`` uses the tetrahedron whose vertices are the three
    basal and one apical N neighbors of that center.
    """

    _validate_request(request)
    trajectory, _ = _trajectory_and_charges(data)
    electric_field = data.electric_field if isinstance(data, ElectrostaticsData) else None
    if request.include_electric_field and electric_field is None:
        raise ValueError(
            "include_electric_field requires electric-field data; for ReaxFF, provide fort.78."
        )
    dipoles = calculate_basal_plane_dipoles(data, request)
    table = dipoles.table.copy()
    factor = float(const("ea3_to_uC_cm2"))
    coordination = (
        _coordination_volumes(dipoles)
        if request.local_volume_method == "coordination"
        else {}
    )

    table["local_volume_method"] = request.local_volume_method
    table["frame_volume_method"] = request.volume_method
    table["frame_volume (angstrom^3)"] = np.nan
    table["valid_center_count"] = 0
    table["local_volume (angstrom^3)"] = np.nan

    for frame, indices in table.groupby("frame_index", sort=False).groups.items():
        frame = int(frame)
        frame_indices = np.asarray(list(indices), dtype=int)
        mu = table.loc[frame_indices, [
            "mu_x (e*angstrom)", "mu_y (e*angstrom)", "mu_z (e*angstrom)"
        ]].to_numpy(float)
        valid_dipole = (
            table.loc[frame_indices, "has_basal_plane_dipole"].to_numpy(bool)
            & np.isfinite(mu).all(axis=1)
        )
        valid_count = int(np.count_nonzero(valid_dipole))
        frame_volume = _frame_volume(trajectory, request, frame)
        table.loc[frame_indices, "frame_volume (angstrom^3)"] = frame_volume
        table.loc[frame_indices, "valid_center_count"] = valid_count
        if request.local_volume_method == "equal":
            if valid_count > 0 and np.isfinite(frame_volume) and frame_volume > 0.0:
                table.loc[frame_indices[valid_dipole], "local_volume (angstrom^3)"] = (
                    frame_volume / valid_count
                )
        else:
            for index in frame_indices[valid_dipole]:
                key = (frame, int(table.at[index, "site_atom_id"]))
                if key in coordination:
                    table.at[index, "local_volume (angstrom^3)"] = coordination[key]

    local_volume = table["local_volume (angstrom^3)"].to_numpy(float)
    table["has_valid_local_volume"] = np.isfinite(local_volume) & (local_volume > 0.0)
    valid_volume = table["has_valid_local_volume"].to_numpy(bool)
    for axis in "xyz":
        mu = table[f"mu_{axis} (e*angstrom)"].to_numpy(float)
        table[f"P_{axis} (uC/cm^2)"] = np.divide(
            mu * factor,
            local_volume,
            out=np.full(len(table), np.nan),
            where=valid_volume & np.isfinite(mu),
        )

    summary_rows: list[dict[str, object]] = []
    for (frame, iteration), group in table.groupby(["frame_index", "iter"], sort=True):
        valid = group["has_valid_local_volume"].to_numpy(bool)
        assigned_volume = float(group.loc[valid, "local_volume (angstrom^3)"].sum())
        row: dict[str, object] = {
            "frame_index": int(frame),
            "iter": int(iteration),
            "center_count": len(group),
            "valid_center_count": int(group["has_basal_plane_dipole"].astype(bool).sum()),
            "valid_local_polarization_count": int(np.count_nonzero(valid)),
            "local_volume_method": request.local_volume_method,
            "frame_volume_method": request.volume_method,
            "frame_volume (angstrom^3)": float(group["frame_volume (angstrom^3)"].iloc[0]),
            "assigned_volume (angstrom^3)": assigned_volume,
        }
        for axis in "xyz":
            mu = float(group.loc[valid, f"mu_{axis} (e*angstrom)"].sum())
            row[f"mu_{axis} (e*angstrom)"] = mu
            row[f"P_{axis} (uC/cm^2)"] = (
                mu / assigned_volume * factor
                if np.isfinite(assigned_volume) and assigned_volume > 0.0
                else np.nan
            )
        summary_rows.append(row)

    poled_counts = directional_poled_counts(
        table,
        trajectory,
        {axis: f"P_{axis} (uC/cm^2)" for axis in "xyz"},
    )
    return BasalPlaneLocalPolarizationResult(
        table=table,
        request=request,
        summary=pd.DataFrame(summary_rows),
        poled_counts=poled_counts,
        dipole_result=dipoles,
        trajectory=trajectory,
        electric_field=electric_field,
        frame_indices=dipoles.frame_indices,
        iterations=dipoles.iterations,
    )


@register_task(
    "get-basal-plane-displacement-local-polarization",
    label="Basal-plane Local Polarization",
)
class BasalPlaneLocalPolarizationTask(AnalysisTask):
    """Calculate per-center polarization using equal or coordination volumes."""

    required_data = TrajectoryData
    supports_selective_streaming = False
    VERSION = "5"

    def required_data_for(
        self, request: BasalPlaneLocalPolarizationRequest, args: dict | None = None
    ):
        if request.include_electric_field:
            return ElectrostaticsData
        return required_wurtzite_data_type(request, args)

    @staticmethod
    def required_data_fields_for(
        request: BasalPlaneLocalPolarizationRequest, _args: dict
    ) -> tuple[str, ...]:
        fields = ["trajectory"]
        if request.charge_source != "formal" or request.include_electric_field:
            fields.append("charges")
        if request.include_electric_field:
            fields.append("electric_field")
        return tuple(fields)

    @staticmethod
    def recommended_presentations(
        _result: BasalPlaneLocalPolarizationResult, _payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        return [
            PresentationSpec(
                renderer="table", label="Basal-plane local polarization", view_type="table"
            )
        ]

    def run(self, data, request: BasalPlaneLocalPolarizationRequest, reporter=None):
        _ = reporter
        return calculate_basal_plane_local_polarization(data, request)


def _table_for_trajectory_frame(
    result: BasalPlaneLocalPolarizationResult,
    frame: int,
) -> tuple[int, pd.DataFrame]:
    """Return source-frame rows for one compact in-memory trajectory frame."""

    source_frame = _source_frame(result.trajectory, int(frame))
    frame_numbers = result.table["frame_index"].astype(int)
    frame_table = result.table[frame_numbers.eq(source_frame)]
    # Direct library calls do not pass through AnalysisExecutor's source-frame
    # restoration, so retain compatibility with their compact-index tables.
    if frame_table.empty and source_frame != int(frame):
        frame_table = result.table[frame_numbers.eq(int(frame))]
    return source_frame, frame_table


def write_local_polarization_extxyz(
    result: BasalPlaneLocalPolarizationResult,
    path: str | Path,
    *,
    precision: int = 8,
) -> Path:
    """Write OVITO-compatible atom properties for every selected frame."""

    trajectory = result.trajectory
    atom_ids = np.asarray(trajectory.atom_ids, dtype=int)
    destination = Path(path)
    with ExtendedXYZWriter(destination, precision=precision) as writer:
        for frame in np.asarray(result.frame_indices, dtype=int):
            positions = np.asarray(trajectory.positions[frame], dtype=float)
            labels = _frame_labels(trajectory, int(frame)).astype(str)
            source_frame, frame_table = _table_for_trajectory_frame(result, int(frame))
            finite = np.isfinite(positions).all(axis=1)
            is_center = np.zeros(len(atom_ids), dtype=int)
            valid_volume = np.zeros(len(atom_ids), dtype=int)
            local_volume = np.full(len(atom_ids), np.nan)
            dipole = np.zeros((len(atom_ids), 3), dtype=float)
            polarization = np.full((len(atom_ids), 3), np.nan)
            id_to_index = {int(atom_id): index for index, atom_id in enumerate(atom_ids)}
            for _, center in frame_table.iterrows():
                atom_index = id_to_index.get(int(center["site_atom_id"]))
                if atom_index is None:
                    continue
                is_center[atom_index] = 1
                valid_volume[atom_index] = int(bool(center["has_valid_local_volume"]))
                local_volume[atom_index] = float(center["local_volume (angstrom^3)"])
                dipole[atom_index] = [
                    float(center[f"mu_{axis} (e*angstrom)"]) for axis in "xyz"
                ]
                polarization[atom_index] = [
                    float(center[f"P_{axis} (uC/cm^2)"]) for axis in "xyz"
                ]
            try:
                lattice = _frame_cell(trajectory, result.request, int(frame))
            except ValueError:
                lattice = None
            iteration = int(frame_table["iter"].iloc[0]) if not frame_table.empty else int(frame)
            metadata: dict[str, object] = {
                "local_volume_method": result.request.local_volume_method,
                "frame_volume_method": result.request.volume_method,
                "dipole_units": "e*angstrom",
                "polarization_units": "uC/cm^2",
                "local_volume_units": "angstrom^3",
            }
            if result.request.include_electric_field:
                if result.electric_field is None:
                    raise ValueError("Electric-field metadata was requested but is unavailable.")
                value = _field_component_series(
                    result.electric_field,
                    component=f"field_{result.request.field_direction}",
                    target_iters=np.asarray([iteration], dtype=int),
                )[0]
                metadata.update({
                    "electric_field": float(value) * float(const("electric_field_VA_to_MVcm")),
                    "electric_field_direction": result.request.field_direction,
                    "electric_field_units": "MV/cm",
                })
            writer.write_frame(
                ExtendedXYZFrame(
                    species=labels[finite].tolist(),
                    positions=positions[finite],
                    properties={
                        "atom_number": atom_ids[finite],
                        "is_local_center": is_center[finite],
                        "has_valid_local_volume": valid_volume[finite],
                        "local_volume": local_volume[finite],
                        "local_dipole": dipole[finite],
                        "local_polarization": polarization[finite],
                    },
                    frame=source_frame,
                    iteration=iteration,
                    lattice=lattice,
                    pbc=(
                        tuple(bool(value) for value in result.request.periodic)
                        if lattice is not None
                        else None
                    ),
                    metadata=metadata,
                )
            )
    return destination


__all__ = [
    "BasalPlaneLocalPolarizationRequest",
    "BasalPlaneLocalPolarizationResult",
    "BasalPlaneLocalPolarizationTask",
    "LocalVolumeMethod",
    "calculate_basal_plane_local_polarization",
    "_table_for_trajectory_frame",
    "write_local_polarization_extxyz",
]
