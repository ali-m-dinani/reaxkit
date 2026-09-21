"""Spatially binned polarization for three-folded wurtzite sites."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.neighbors import (
    _frame_cell,
    _trajectory_and_charges,
    required_wurtzite_data_type,
)
from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.polarity import (
    WurtzitePolarityRequest,
    WurtzitePolarityResult,
    calculate_polarity_from_trajectory,
)
from reaxkit.core.platform.constants import const
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import TrajectoryData
from reaxkit.presentation.specs import PresentationSpec

VolumeMethod = Literal["hull", "bbox", "cell"]
AXES = ("x", "y", "z")


@dataclass
class BinnedPolarizationRequest(WurtzitePolarityRequest):
    """Configure spatial bins and their polarization-normalization volume."""

    bins_x: int = dc_field(default=1, metadata={"label": "X bins", "min": 1})
    bins_y: int = dc_field(default=1, metadata={"label": "Y bins", "min": 1})
    bins_z: int = dc_field(default=1, metadata={"label": "Z bins", "min": 1})
    volume_method: VolumeMethod = dc_field(
        default="hull",
        metadata={"label": "Volume method", "choices": ["hull", "bbox", "cell"]},
    )


@dataclass
class BinnedPolarizationResult(BaseResult):
    """Dense per-frame bin table calculated from site dipole moments."""

    table: pd.DataFrame
    request: BinnedPolarizationRequest
    summary: pd.DataFrame
    polarity_result: WurtzitePolarityResult
    bin_edges: tuple[np.ndarray, np.ndarray, np.ndarray]
    frame_indices: np.ndarray
    iterations: np.ndarray

    @property
    def csv_tables(self) -> dict[str, pd.DataFrame]:
        return {"binned_polarization": self.table, "polarization_summary": self.summary}


def _validate_request(request: BinnedPolarizationRequest) -> tuple[int, int, int]:
    bins = (int(request.bins_x), int(request.bins_y), int(request.bins_z))
    if any(value < 1 for value in bins):
        raise ValueError("bins_x, bins_y, and bins_z must each be at least 1.")
    if request.volume_method not in {"hull", "bbox", "cell"}:
        raise ValueError("volume_method must be one of: hull, bbox, cell.")
    return bins


def _axis_edges(values: np.ndarray, count: int) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("The reference frame has no finite coordinates.")
    lower, upper = float(np.min(finite)), float(np.max(finite))
    if lower == upper:
        padding = max(abs(lower) * 0.05, 0.5)
        lower, upper = lower - padding, upper + padding
    # Include coordinates equal to the upper edge without special cases in plots.
    return np.linspace(lower, np.nextafter(upper, np.inf), count + 1)


def _bin_indices(coords: np.ndarray, edges: tuple[np.ndarray, ...]) -> np.ndarray:
    shape = tuple(len(edge) - 1 for edge in edges)
    indices = []
    for axis, edge in enumerate(edges):
        values = np.searchsorted(edge, coords[:, axis], side="right") - 1
        indices.append(np.clip(values, 0, shape[axis] - 1))
    return np.ravel_multi_index(tuple(indices), shape)


def _bbox_volume(coords: np.ndarray) -> float:
    if len(coords) == 0:
        return np.nan
    extents = np.ptp(np.asarray(coords, dtype=float), axis=0)
    return float(np.prod(extents)) if np.isfinite(extents).all() else np.nan


def _hull_volume(coords: np.ndarray) -> float:
    if len(coords) < 4:
        return np.nan
    try:
        return float(ConvexHull(np.asarray(coords, dtype=float)).volume)
    except Exception:
        return np.nan


def _cell_bin_volume(data: TrajectoryData, request: BinnedPolarizationRequest, frame: int, bins: tuple[int, int, int]) -> float:
    cell = _frame_cell(data, request, frame)
    if cell is None:
        return np.nan
    volume = abs(float(np.linalg.det(np.asarray(cell, dtype=float))))
    return volume / float(np.prod(bins)) if np.isfinite(volume) and volume > 0 else np.nan


def _metadata(edges: tuple[np.ndarray, ...], bins: tuple[int, int, int]) -> pd.DataFrame:
    flat = np.arange(int(np.prod(bins)), dtype=int)
    indices = np.unravel_index(flat, bins)
    values: dict[str, np.ndarray] = {"bin_number": flat + 1}
    for axis_index, axis in enumerate(AXES):
        index = indices[axis_index]
        edge = edges[axis_index]
        values[f"bin_{axis}"] = index
        values[f"{axis}_min"] = edge[index]
        values[f"{axis}_max"] = edge[index + 1]
        values[f"{axis}_center"] = (edge[index] + edge[index + 1]) / 2.0
    return pd.DataFrame(values)


def _frame_rows(
    trajectory: TrajectoryData,
    sites: pd.DataFrame,
    *,
    frame: int,
    iteration: int,
    request: BinnedPolarizationRequest,
    bins: tuple[int, int, int],
    edges: tuple[np.ndarray, ...],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    output = metadata.copy()
    output.insert(0, "iter", int(iteration))
    output.insert(0, "frame_index", int(frame))
    n_bins = len(output)

    site_coords = sites[[
        "site_x (angstrom)", "site_y (angstrom)", "site_z (angstrom)"
    ]].to_numpy(float)
    site_bins = _bin_indices(site_coords, edges) if len(site_coords) else np.empty(0, int)
    complete = sites["has_three_basal_neighbors"].to_numpy(bool)
    dipoles = sites[[
        "p_x (e*angstrom)", "p_y (e*angstrom)", "p_z (e*angstrom)"
    ]].to_numpy(float)
    valid = complete & np.isfinite(dipoles).all(axis=1)
    output["site_count"] = np.bincount(site_bins, minlength=n_bins)
    output["valid_dipole_count"] = np.bincount(site_bins[valid], minlength=n_bins)

    frame_coords = np.asarray(trajectory.positions[frame], dtype=float)
    frame_coords = frame_coords[np.isfinite(frame_coords).all(axis=1)]
    atom_bins = _bin_indices(frame_coords, edges) if len(frame_coords) else np.empty(0, int)
    output["atom_count"] = np.bincount(atom_bins, minlength=n_bins)

    for component, axis in enumerate(AXES):
        sums = np.bincount(
            site_bins[valid], weights=dipoles[valid, component], minlength=n_bins
        )
        output[f"mu_{axis} (e*angstrom)"] = sums
        output[f"mu_{axis} (debye)"] = sums * float(const("ea_to_debye"))

    if request.volume_method == "cell":
        volumes = np.full(n_bins, _cell_bin_volume(trajectory, request, frame, bins))
    else:
        estimator = _hull_volume if request.volume_method == "hull" else _bbox_volume
        volumes = np.asarray([
            estimator(frame_coords[atom_bins == bin_index]) for bin_index in range(n_bins)
        ])
    output["volume_method"] = request.volume_method
    output["volume (angstrom^3)"] = volumes
    factor = float(const("ea3_to_uC_cm2"))
    for axis in AXES:
        dipole = output[f"mu_{axis} (e*angstrom)"].to_numpy(float)
        output[f"P_{axis} (uC/cm^2)"] = np.divide(
            dipole * factor,
            volumes,
            out=np.full(n_bins, np.nan),
            where=np.isfinite(volumes) & (volumes > 0.0),
        )
    return output


def calculate_binned_polarization(data, request: BinnedPolarizationRequest) -> BinnedPolarizationResult:
    """Sum existing local dipoles in spatial bins and divide by bin volume."""

    bins = _validate_request(request)
    trajectory, _ = _trajectory_and_charges(data)
    polarity = calculate_polarity_from_trajectory(data, request)
    positions = np.asarray(trajectory.positions, dtype=float)
    reference = positions[int(request.reference_frame)]
    reference = reference[np.isfinite(reference).all(axis=1)]
    edges = tuple(_axis_edges(reference[:, axis], bins[axis]) for axis in range(3))
    metadata = _metadata(edges, bins)
    rows: list[pd.DataFrame] = []
    for frame, sites in polarity.table.groupby("frame_index", sort=True):
        iteration = int(sites["iter"].iloc[0])
        rows.append(_frame_rows(
            trajectory, sites, frame=int(frame), iteration=iteration, request=request,
            bins=bins, edges=edges, metadata=metadata,
        ))
    table = pd.concat(rows, ignore_index=True) if rows else metadata.iloc[0:0].copy()
    summary_rows = []
    for (frame, iteration), group in table.groupby(["frame_index", "iter"], sort=True):
        volume = float(group["volume (angstrom^3)"].sum(min_count=1))
        row: dict[str, object] = {
            "frame_index": int(frame), "iter": int(iteration),
            "site_count": int(group["site_count"].sum()),
            "valid_dipole_count": int(group["valid_dipole_count"].sum()),
            "volume_method": request.volume_method, "volume (angstrom^3)": volume,
        }
        for axis in AXES:
            mu = float(group[f"mu_{axis} (e*angstrom)"].sum())
            row[f"mu_{axis} (e*angstrom)"] = mu
            row[f"P_{axis} (uC/cm^2)"] = (
                mu / volume * float(const("ea3_to_uC_cm2"))
                if np.isfinite(volume) and volume > 0 else np.nan
            )
        summary_rows.append(row)
    return BinnedPolarizationResult(
        table=table, request=request, summary=pd.DataFrame(summary_rows),
        polarity_result=polarity, bin_edges=edges,
        frame_indices=polarity.frame_indices, iterations=polarity.iterations,
    )


@register_task(
    "get-three-folded-wurtzite-polarization",
    label="Three-folded Wurtzite Polarization",
)
class BinnedPolarizationTask(AnalysisTask):
    """Calculate spatially binned polarization from three-folded site dipoles."""

    required_data = TrajectoryData
    supports_selective_streaming = False
    VERSION = "1"

    def required_data_for(self, request: BinnedPolarizationRequest, args: dict | None = None):
        return required_wurtzite_data_type(request, args)

    @staticmethod
    def required_data_fields_for(request: BinnedPolarizationRequest, _args: dict) -> tuple[str, ...]:
        return ("trajectory", "charges") if request.charge_source != "formal" else ("trajectory",)

    @staticmethod
    def recommended_presentations(
        _result: BinnedPolarizationResult, _payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        return [PresentationSpec(renderer="table", label="Binned polarization", view_type="table")]

    def run(self, data, request: BinnedPolarizationRequest, reporter=None):
        _ = reporter
        return calculate_binned_polarization(data, request)


__all__ = [
    "AXES", "BinnedPolarizationRequest", "BinnedPolarizationResult",
    "BinnedPolarizationTask", "VolumeMethod", "calculate_binned_polarization",
]
