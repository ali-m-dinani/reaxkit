"""Binned polarization from fixed-anion basal-plane contributions.

The reference-plane construction follows Hayden et al., "Ferroelectricity in
boron-substituted aluminum nitride thin films," Physical Review Materials 5,
044412 (2021), https://doi.org/10.1103/PhysRevMaterials.5.044412. This code uses
formal or ReaxFF ``fort.7`` atomic charges in place of Born effective charges.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any

import numpy as np
import pandas as pd

from reaxkit.core.runtime.execution_contracts import TaskCapabilities, ExecutionShape
from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.dipole import (
    BasalPlaneDipoleRequest,
    BasalPlaneDipoleResult,
    calculate_basal_plane_dipoles,
)
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.neighbors import (
    _trajectory_and_charges,
    required_wurtzite_data_type,
)
from reaxkit.analysis.ferroelectrics.poled_counts import directional_poled_counts
from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.polarization import (
    VolumeMethod,
    _axis_edges,
    _bbox_volume,
    _bin_indices,
    _cell_bin_volume,
    _hull_volume,
    _metadata,
)
from reaxkit.core.platform.constants import const
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import TrajectoryData
from reaxkit.presentation.specs import PresentationSpec


@dataclass
class BasalPlanePolarizationRequest(BasalPlaneDipoleRequest):
    """Configure spatial polarization from Hayden et al.-style local dipoles."""

    bins_x: int = dc_field(default=1, metadata={"label": "X bins", "min": 1})
    bins_y: int = dc_field(default=1, metadata={"label": "Y bins", "min": 1})
    bins_z: int = dc_field(default=1, metadata={"label": "Z bins", "min": 1})
    volume_method: VolumeMethod = dc_field(
        default="hull",
        metadata={"label": "Volume method", "choices": ["hull", "bbox", "cell"]},
    )


@dataclass
class BasalPlanePolarizationResult(BaseResult):
    """Per-bin polarization and the basal-plane dipoles used to calculate it."""

    table: pd.DataFrame
    request: BasalPlanePolarizationRequest
    summary: pd.DataFrame
    poled_counts: pd.DataFrame
    dipole_result: BasalPlaneDipoleResult
    bin_edges: tuple[np.ndarray, np.ndarray, np.ndarray]
    frame_indices: np.ndarray
    iterations: np.ndarray

    @property
    def csv_tables(self) -> dict[str, pd.DataFrame]:
        return {
            "basal_plane_polarization": self.table,
            "basal_plane_polarization_summary": self.summary,
            "basal_plane_polarization_poled_counts": self.poled_counts,
        }


def _validate_request(request: BasalPlanePolarizationRequest) -> tuple[int, int, int]:
    bins = (int(request.bins_x), int(request.bins_y), int(request.bins_z))
    if any(value < 1 for value in bins):
        raise ValueError("bins_x, bins_y, and bins_z must each be at least 1.")
    if request.volume_method not in {"hull", "bbox", "cell"}:
        raise ValueError("volume_method must be one of: hull, bbox, cell.")
    return bins


def calculate_basal_plane_polarization(
    data,
    request: BasalPlanePolarizationRequest,
    dipoles=None, reference_positions=None,
) -> BasalPlanePolarizationResult:
    """Calculate ``P = sum(mu_ion) / volume`` with each supercell ion counted once.

    The displacement reference follows Hayden et al., "Ferroelectricity in
    boron-substituted aluminum nitride thin films," Physical Review Materials
    5, 044412 (2021), https://doi.org/10.1103/PhysRevMaterials.5.044412.
    The local approximation fixes the anion framework and displaces each
    center once from its own basal plane; it does not use an apical bond height
    as an additional ionic displacement.
    """

    bins = _validate_request(request)
    trajectory, _ = _trajectory_and_charges(data)
    dipoles = dipoles if dipoles is not None else calculate_basal_plane_dipoles(data, request)
    positions = np.asarray(trajectory.positions, dtype=float)
    reference = positions[int(request.reference_frame)] if reference_positions is None else reference_positions
    reference = reference[np.isfinite(reference).all(axis=1)]
    edges = tuple(_axis_edges(reference[:, axis], bins[axis]) for axis in range(3))
    metadata = _metadata(edges, bins)
    factor = float(const("ea3_to_uC_cm2"))
    rows: list[pd.DataFrame] = []

    for frame, ions in dipoles.ions.groupby("frame_index", sort=True):
        frame = int(frame)
        output = metadata.copy()
        output.insert(0, "iter", int(ions["iter"].iloc[0]))
        output.insert(0, "frame_index", frame)
        n_bins = len(output)
        ion_coords = ions[[
            "x (angstrom)", "y (angstrom)", "z (angstrom)"
        ]].to_numpy(float)
        ion_bins = _bin_indices(ion_coords, edges)
        local_mu = ions[[
            "mu_x (e*angstrom)", "mu_y (e*angstrom)", "mu_z (e*angstrom)"
        ]].to_numpy(float)
        valid = ions["has_ion_dipole"].to_numpy(bool) & np.isfinite(local_mu).all(axis=1)
        output["ion_count"] = np.bincount(ion_bins, minlength=n_bins)
        output["valid_ion_count"] = np.bincount(ion_bins[valid], minlength=n_bins)

        frame_coords = positions[frame]
        frame_coords = frame_coords[np.isfinite(frame_coords).all(axis=1)]
        atom_bins = _bin_indices(frame_coords, edges)
        output["atom_count"] = np.bincount(atom_bins, minlength=n_bins)
        for component, axis in enumerate("xyz"):
            values = np.bincount(
                ion_bins[valid], weights=local_mu[valid, component], minlength=n_bins
            )
            output[f"mu_{axis} (e*angstrom)"] = values
            output[f"mu_{axis} (debye)"] = values * float(const("ea_to_debye"))

        if request.volume_method == "cell":
            volumes = np.full(n_bins, _cell_bin_volume(trajectory, request, frame, bins))
        else:
            estimator = _hull_volume if request.volume_method == "hull" else _bbox_volume
            volumes = np.asarray([
                estimator(frame_coords[atom_bins == index]) for index in range(n_bins)
            ])
        output["volume_method"] = request.volume_method
        output["volume (angstrom^3)"] = volumes
        for axis in "xyz":
            mu = output[f"mu_{axis} (e*angstrom)"].to_numpy(float)
            output[f"P_{axis} (uC/cm^2)"] = np.divide(
                mu * factor, volumes, out=np.full(n_bins, np.nan),
                where=np.isfinite(volumes) & (volumes > 0.0),
            )
        rows.append(output)

    table = pd.concat(rows, ignore_index=True) if rows else metadata.iloc[0:0].copy()
    summary_rows: list[dict[str, object]] = []
    for (frame, iteration), group in table.groupby(["frame_index", "iter"], sort=True):
        volume = float(group["volume (angstrom^3)"].sum(min_count=1))
        row: dict[str, object] = {
            "frame_index": int(frame), "iter": int(iteration),
            "ion_count": int(group["ion_count"].sum()),
            "valid_ion_count": int(group["valid_ion_count"].sum()),
            "volume_method": request.volume_method, "volume (angstrom^3)": volume,
        }
        for axis in "xyz":
            mu = float(group[f"mu_{axis} (e*angstrom)"].sum())
            row[f"mu_{axis} (e*angstrom)"] = mu
            row[f"P_{axis} (uC/cm^2)"] = (
                mu / volume * factor if np.isfinite(volume) and volume > 0 else np.nan
            )
        summary_rows.append(row)
    poled_counts = directional_poled_counts(
        table,
        trajectory,
        {axis: f"P_{axis} (uC/cm^2)" for axis in "xyz"},
    )
    return BasalPlanePolarizationResult(
        table=table, request=request, summary=pd.DataFrame(summary_rows),
        poled_counts=poled_counts,
        dipole_result=dipoles, bin_edges=edges,
        frame_indices=dipoles.frame_indices, iterations=dipoles.iterations,
    )


@register_task(
    "get-basal-plane-displacement-polarization",
    label="Basal-plane Displacement Polarization",
)
class BasalPlanePolarizationTask(AnalysisTask):
    """Normalize Hayden et al.-style basal-plane dipoles by spatial-bin volume."""

    required_data = TrajectoryData
    supports_output_profiles = True
    execution_capabilities = TaskCapabilities(shape=ExecutionShape.REFERENCE_FRAME_MAP,
        thread_safe=True, automatic_parallel=False, supports_selective_frames=True,
        reference_fields=("reference_frame",), estimated_frame_bytes=16 * 1024 * 1024)
    supports_selective_streaming = False
    VERSION = "3"

    def required_data_for(self, request: BasalPlanePolarizationRequest, args: dict | None = None):
        return required_wurtzite_data_type(request, args)

    @staticmethod
    def required_data_fields_for(request: BasalPlanePolarizationRequest, _args: dict) -> tuple[str, ...]:
        return ("trajectory", "charges") if request.charge_source != "formal" else ("trajectory",)

    @staticmethod
    def recommended_presentations(
        _result: BasalPlanePolarizationResult, _payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        return [PresentationSpec(renderer="table", label="Basal-plane polarization", view_type="table")]

    def run(self, data, request: BasalPlanePolarizationRequest, reporter=None):
        _ = reporter
        return calculate_basal_plane_polarization(data, request)

    def run_stream(self, frames, request, reporter=None, pipeline=None):
        from reaxkit.analysis.ferroelectrics.polarization_stream import stream_polarization
        return stream_polarization(self, frames, request, "basal_binned", reporter, pipeline)


__all__ = [
    "BasalPlanePolarizationRequest", "BasalPlanePolarizationResult",
    "BasalPlanePolarizationTask", "calculate_basal_plane_polarization",
]
