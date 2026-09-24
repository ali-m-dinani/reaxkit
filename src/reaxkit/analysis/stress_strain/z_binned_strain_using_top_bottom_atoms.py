"""Compute robust z-binned geometric strain from top/bottom atom spans.

The analyzer compares per-bin coordinate spans against frame zero. Each span
uses averages of the largest and smallest selected coordinates, which reduces
sensitivity to a single extreme atom. File loading and output rendering remain
workflow responsibilities.

**Usage context**

- Slab deformation: Track normal strain profiles through material thickness.
- Robust geometry: Reduce sensitivity to noisy extrema with atom-set averages.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.core.runtime.execution_contracts import TaskCapabilities, ExecutionShape
from reaxkit.analysis.stress_strain.streaming import selected_coordinates, run_strain_stream
from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis.stress_strain import common
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import TrajectoryData
from reaxkit.presentation.specs import PresentationSpec

SPAN_COLUMNS = ("dx", "dy", "dz")
STRAIN_COLUMNS = ("strain_xx", "strain_yy", "strain_zz")
RESULT_COLUMNS = [
    "frame", "iter", "bin_number", "bin_mean_z", "number_of_atoms_in_this_bin",
    "mean_max_x", "mean_min_x", "dx", "dx_frame_0", "span_change_x", "strain_xx",
    "mean_max_y", "mean_min_y", "dy", "dy_frame_0", "span_change_y", "strain_yy",
    "mean_max_z", "mean_min_z", "dz", "dz_frame_0", "span_change_z", "strain_zz",
]


@dataclass
class ZBinnedTopBottomStrainRequest(BaseRequest):
    """Request configuration for robust z-binned span strain.

    Fields
    -----
    z_bins : int
        Number of equal-width z bins; must be at least one. Default is `20`.
    atom_types : Optional[Sequence[str]]
        Elements to include; `None` selects all atoms. Default is `(\"Al\", \"N\")`.
    bin_range : str
        Use `\"reference\"` for fixed frame-zero bins or `\"current\"` for
        frame-local bins. Default is `\"reference\"`.
    n_extreme_atoms : int
        Number of atoms averaged at each coordinate extreme. Default is `5`.
    selected_frames : Optional[Sequence[int]]
        Zero-based frames to report; `None` reports all frames.
    every : int
        Stride applied to the frame selection. Default is `1`.
    unwrap : bool
        Cumulatively unwrap periodic coordinates. Default is `True`.
    periodic : str
        Periodic axes such as `\"xy\"`, `\"xyz\"`, or `\"none\"`.
    zero_tolerance : float
        Baseline spans at or below this magnitude produce undefined strain.

    Examples
    -----
    `ZBinnedTopBottomStrainRequest(z_bins=20, atom_types=(\"Al\", \"N\"), periodic=\"xy\")`
    This configures fixed reference bins for an xy-periodic slab.
    """

    z_bins: int = dc_field(default=20, metadata={"label": "Z bins", "help": "Number of equal-width z bins.", "min": 1})
    atom_types: Optional[Sequence[str]] = dc_field(default=("Al", "N"), metadata={"label": "Atom types", "help": "Elements to include; empty selects all atoms."})
    bin_range: str = dc_field(default="reference", metadata={"label": "Bin range", "help": "Fixed reference or current-frame z bins.", "choices": ["reference", "current"]})
    n_extreme_atoms: int = dc_field(default=5, metadata={"label": "Top/bottom count", "help": "Atoms averaged at each coordinate extreme.", "min": 1})
    selected_frames: Optional[Sequence[int]] = dc_field(default=None, metadata={"label": "Frames", "help": "Zero-based frames to report."})
    every: int = dc_field(default=1, metadata={"label": "Frame stride", "help": "Stride over selected frames.", "min": 1})
    unwrap: bool = dc_field(default=True, metadata={"label": "Unwrap PBC", "help": "Cumulatively unwrap periodic coordinates."})
    periodic: str = dc_field(default="xyz", metadata={"label": "Periodic axes", "help": "Periodic axes, such as xy or xyz.", "choices": ["none", "x", "y", "z", "xy", "xz", "yz", "xyz"]})
    zero_tolerance: float = dc_field(default=1.0e-12, metadata={"label": "Zero tolerance", "help": "Smallest usable reference span.", "min": 0.0})


@dataclass
class ZBinnedTopBottomStrainResult(BaseResult):
    """Result of robust z-binned span strain analysis.

    Fields
    -----
    table : pd.DataFrame
        One row per reported frame and bin with extrema, spans, span changes,
        and fractional normal strains.
    request : ZBinnedTopBottomStrainRequest
        Request that produced the table.

    Examples
    -----
    `ZBinnedTopBottomStrainResult(table=<DataFrame>, request=request)` stores
    both computed values and their provenance.
    """

    table: pd.DataFrame
    request: ZBinnedTopBottomStrainRequest


def _span_rows(coordinates: np.ndarray, bins: np.ndarray, edges: np.ndarray, *, frame: int, iteration: int, count: int) -> pd.DataFrame:
    """Build raw extreme averages and spans for one frame."""
    rows: list[dict[str, int | float]] = []
    for zero_based_bin, midpoint in enumerate((edges[:-1] + edges[1:]) / 2.0):
        bin_number = zero_based_bin + 1
        values = coordinates[bins == bin_number]
        atom_count = int(values.shape[0])
        row: dict[str, int | float] = {
            "frame": frame, "iter": iteration, "bin_number": bin_number,
            "bin_mean_z": float(midpoint), "number_of_atoms_in_this_bin": atom_count,
        }
        for axis_name, axis in zip("xyz", range(3)):
            if atom_count >= 2 * count:
                ordered = np.sort(values[:, axis])
                mean_min = float(np.mean(ordered[:count]))
                mean_max = float(np.mean(ordered[-count:]))
                span = mean_max - mean_min
            else:
                mean_min = mean_max = span = float("nan")
            row[f"mean_max_{axis_name}"] = mean_max
            row[f"mean_min_{axis_name}"] = mean_min
            row[f"d{axis_name}"] = span
        rows.append(row)
    return pd.DataFrame(rows)


def calculate_top_bottom_strain(data: TrajectoryData, request: ZBinnedTopBottomStrainRequest, *, stream=None) -> pd.DataFrame:
    """Calculate robust span changes and fractional normal strain.

    Works on
    -----
    `TrajectoryData` plus `ZBinnedTopBottomStrainRequest` analyzer inputs.

    Parameters
    -----
    data : TrajectoryData
        Canonical positions, identities, iterations, and optional cell data.
    request : ZBinnedTopBottomStrainRequest
        Atom, bin, frame, and periodic-boundary controls.

    Returns
    -----
    pd.DataFrame
        Per-frame/per-bin span and strain rows in `RESULT_COLUMNS` order.

    Examples
    -----
    ```python
    table = calculate_top_bottom_strain(data, ZBinnedTopBottomStrainRequest(z_bins=10, unwrap=False))
    print(table[["bin_number", "strain_xx"]].head())
    ```
    Sample output contains one normal-strain value per z bin; zero is unchanged.
    """
    positions = np.asarray(data.positions, dtype=float)
    if request.n_extreme_atoms < 1:
        raise ValueError("n_extreme_atoms must be at least 1.")
    if request.zero_tolerance < 0.0 or not np.isfinite(request.zero_tolerance):
        raise ValueError("zero_tolerance must be finite and nonnegative.")
    bin_range = str(request.bin_range).lower()
    if bin_range not in {"reference", "current"}:
        raise ValueError("bin_range must be 'reference' or 'current'.")
    frames = (common.selected_frames(positions.shape[0], request.selected_frames, request.every) if stream is None
              else None if request.selected_frames is None else sorted(request.selected_frames)[::request.every])
    frames_needed = sorted({0, *frames}) if frames is not None else None
    selected = common.select_atom_indices(data.elements, request.atom_types)
    reference_finite = np.all(np.isfinite(positions[0, selected]), axis=1)
    eligible = selected[reference_finite]
    if eligible.size == 0:
        raise ValueError("No selected atoms have finite coordinates in frame 0.")
    reference_coordinates = positions[0, eligible]
    reference_edges = common.bin_edges(reference_coordinates[:, 2], request.z_bins)
    reference_bins = common.assign_bins(reference_coordinates[:, 2], reference_edges)
    iterations = common.iteration_values(data)
    baseline: pd.DataFrame | None = None
    outputs: list[pd.DataFrame] = []
    coordinate_rows = (selected_coordinates(stream[0], eligible, request, stream[1], include_reference=True) if stream is not None else
                       ((frame, coords, valid, int(iterations[frame])) for frame, coords, valid in common.iter_selected_coordinates(data, eligible, frames_needed, unwrap=request.unwrap, periodic=request.periodic)))
    for frame, all_coordinates, valid, iteration in coordinate_rows:
        coordinates = all_coordinates[valid]
        if coordinates.size == 0:
            raise ValueError(f"Frame {frame} contains no finite selected atoms.")
        edges = reference_edges if bin_range == "reference" else common.bin_edges(coordinates[:, 2], request.z_bins)
        bins = reference_bins[valid] if bin_range == "reference" else common.assign_bins(coordinates[:, 2], edges)
        raw = _span_rows(coordinates, bins, edges, frame=frame, iteration=iteration, count=request.n_extreme_atoms)
        if frame == 0:
            baseline = raw[["bin_number", *SPAN_COLUMNS]].rename(columns={name: f"{name}_frame_0" for name in SPAN_COLUMNS})
        if (frames is not None and frame not in frames) or (frames is None and frame % request.every):
            continue
        assert baseline is not None
        result = raw.merge(baseline, on="bin_number", how="left", validate="one_to_one")
        for axis in "xyz":
            span, base = f"d{axis}", f"d{axis}_frame_0"
            result[f"span_change_{axis}"] = result[span] - result[base]
            result[f"strain_{axis}{axis}"] = np.nan
            usable = np.abs(result[base]) > request.zero_tolerance
            result.loc[usable, f"strain_{axis}{axis}"] = result.loc[usable, f"span_change_{axis}"] / result.loc[usable, base]
        outputs.append(result[RESULT_COLUMNS])
    if not outputs:
        raise ValueError("No requested frames were analyzed.")
    return pd.concat(outputs, ignore_index=True)[RESULT_COLUMNS]


@register_task("get_z_binned_top_bottom_strain", label="Z-Binned Top/Bottom Strain")
class ZBinnedTopBottomStrainTask(AnalysisTask):
    """Run robust top/bottom span strain analysis on canonical trajectories."""

    required_data = TrajectoryData
    execution_capabilities = TaskCapabilities(shape=ExecutionShape.ORDERED_STATEFUL_STREAM,
        supports_selective_frames=True, reference_frames=(0,), requires_contiguous_history=True,
        estimated_frame_bytes=8 * 1024 * 1024)

    def run_stream(self, frames, request, reporter=None, pipeline=None):
        return run_strain_stream(self, frames, request, calculate_top_bottom_strain, ZBinnedTopBottomStrainResult, pipeline)

    @staticmethod
    def recommended_presentations(_result: ZBinnedTopBottomStrainResult, payload: dict[str, Any]) -> list[PresentationSpec]:
        """Return table and normal-strain profile presentation specifications.

        Works on
        -----
        Analyzer task output payloads.

        Parameters
        -----
        _result : ZBinnedTopBottomStrainResult
            Typed analyzer result.
        payload : dict[str, Any]
            Serialized result payload.

        Returns
        -----
        list[PresentationSpec]
            Table and strain-versus-z plot specifications.

        Examples
        -----
        `ZBinnedTopBottomStrainTask.recommended_presentations(result, payload)`
        returns standard views for downstream presentation dispatch.
        """
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(renderer="single_plot", label="Normal Strain vs Z", mapping={"x_col": "bin_mean_z", "y_col": "strain_zz", "group_by_col": "frame"}, options={"title": "Z-Binned Normal Strain", "xlabel": "Bin mean z", "ylabel": "strain_zz", "legend": True}, view_type="plot2d"),
        ]

    def run(self, data: TrajectoryData, request: ZBinnedTopBottomStrainRequest, reporter=None) -> ZBinnedTopBottomStrainResult:
        """Execute robust top/bottom span strain analysis.

        Works on
        -----
        `TrajectoryData` plus `ZBinnedTopBottomStrainRequest` analyzer inputs.

        Parameters
        -----
        data : TrajectoryData
            Canonical trajectory data.
        request : ZBinnedTopBottomStrainRequest
            Analysis configuration.
        reporter : Any, optional
            Progress reporter accepted by the analyzer interface.

        Returns
        -----
        ZBinnedTopBottomStrainResult
            Computed table and request provenance.

        Examples
        -----
        `ZBinnedTopBottomStrainTask().run(data, request)` returns one row per
        selected frame and z bin.
        """
        _ = reporter
        return ZBinnedTopBottomStrainResult(table=calculate_top_bottom_strain(data, request), request=request)
