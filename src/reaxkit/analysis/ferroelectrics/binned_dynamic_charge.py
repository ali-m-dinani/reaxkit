"""Sum dynamic atomic charges in a frame-zero-fixed two-dimensional grid."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.core.runtime.execution_contracts import TaskCapabilities, ExecutionShape
from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ElectrostaticsData
from reaxkit.presentation.specs import PresentationSpec

Plane = Literal["xy", "xz", "yz"]
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
BIN_COLUMNS = [
    "frame", "iteration", "bin_number", "bin_x", "bin_y", "bin_z",
    "x_min", "x_max", "x_center", "y_min", "y_max", "y_center",
    "z_min", "z_max", "z_center", "atom_count", "valid_charge_count",
    "charge", "delta_charge",
]
AVERAGE_COLUMNS = ["average_charge", "average_delta_charge"]
HEATMAP_CMAP = "coolwarm_r"


@dataclass
class BinnedDynamicChargeRequest(BaseRequest):
    """Configure a projected charge grid fixed by frame-zero coordinates.

    Only the two bin counts belonging to ``plane`` are required. For example,
    ``plane="xz"`` uses ``bins_x`` and ``bins_z`` and sums over all y values.
    """

    plane: Plane = dc_field(
        default="xz",
        metadata={"label": "Plane", "choices": ["xy", "xz", "yz"]},
    )
    bins_x: Optional[int] = dc_field(
        default=None,
        metadata={"label": "X bins", "help": "Required for xy and xz.", "min": 1},
    )
    bins_y: Optional[int] = dc_field(
        default=None,
        metadata={"label": "Y bins", "help": "Required for xy and yz.", "min": 1},
    )
    bins_z: Optional[int] = dc_field(
        default=None,
        metadata={"label": "Z bins", "help": "Required for xz and yz.", "min": 1},
    )
    selected_frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={"label": "Frames", "help": "Optional source-frame indices."},
    )
    every: int = dc_field(
        default=1,
        metadata={"label": "Every", "help": "Keep every Nth selected frame.", "min": 1},
    )
    average: bool = dc_field(
        default=False,
        metadata={
            "label": "Average per bin",
            "help": "Add per-particle averages and plot averages instead of sums.",
        },
    )
    _expected_frames: Optional[int] = None


@dataclass
class BinnedDynamicChargeResult(BaseResult):
    """Dense per-frame projected-bin table and immutable frame-zero grid."""

    table: pd.DataFrame
    request: BinnedDynamicChargeRequest
    frame_indices: np.ndarray
    iterations: np.ndarray
    plane: Plane
    axis_labels: tuple[str, str]
    bin_edges: tuple[np.ndarray, np.ndarray]
    atom_bin_numbers: dict[int, int]
    time_values: Optional[np.ndarray] = None

    @property
    def csv_tables(self) -> dict[str, pd.DataFrame]:
        return {"binned_dynamic_charge": self.table}


def _plane_configuration(
    request: BinnedDynamicChargeRequest,
) -> tuple[Plane, tuple[str, str], tuple[int, int]]:
    plane = str(request.plane).lower()
    if plane not in {"xy", "xz", "yz"}:
        raise ValueError("plane must be one of: xy, xz, yz.")
    axis_labels = (plane[0], plane[1])
    counts_by_axis = {"x": request.bins_x, "y": request.bins_y, "z": request.bins_z}
    missing = [axis for axis in axis_labels if counts_by_axis[axis] is None]
    if missing:
        required = ", ".join(f"bins_{axis}" for axis in missing)
        raise ValueError(f"The {plane} plane requires {required}.")
    bins = tuple(int(counts_by_axis[axis]) for axis in axis_labels)
    if any(value < 1 for value in bins):
        raise ValueError("Plane bin counts must each be at least 1.")
    if int(request.every) < 1:
        raise ValueError("every must be at least 1.")
    return plane, axis_labels, bins


def _selected_frames(n_frames: int, request: BinnedDynamicChargeRequest) -> np.ndarray:
    if n_frames < 1:
        raise ValueError("At least one frame is required so frame 0 can define the bins.")
    requested = (
        np.arange(n_frames, dtype=int)
        if request.selected_frames is None
        else np.asarray(request.selected_frames, dtype=int)
    )
    if requested.ndim != 1:
        raise ValueError("selected_frames must be a one-dimensional sequence.")
    if np.any(requested < 0) or np.any(requested >= n_frames):
        raise ValueError(f"selected_frames must be between 0 and {n_frames - 1}.")
    return requested[:: int(request.every)]


def _axis_edges(values: np.ndarray, number_of_bins: int) -> np.ndarray:
    lower = float(np.min(values))
    upper = float(np.max(values))
    if lower == upper:
        padding = max(abs(lower) * 0.05, 0.5)
        lower -= padding
        upper += padding
    return np.linspace(lower, upper, number_of_bins + 1, dtype=float)


def _fixed_grid(
    positions: np.ndarray,
    atom_ids: np.ndarray,
    axis_labels: tuple[str, str],
    bins: tuple[int, int],
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    finite = np.isfinite(positions).all(axis=1)
    if not np.any(finite):
        raise ValueError("Frame 0 must contain at least one atom with finite coordinates.")
    selected_positions = positions[finite]
    selected_ids = atom_ids[finite]
    coordinate_indices = tuple(AXIS_INDEX[axis] for axis in axis_labels)
    edges = tuple(
        _axis_edges(selected_positions[:, coordinate_index], bins[plane_index])
        for plane_index, coordinate_index in enumerate(coordinate_indices)
    )
    bin_indices = []
    for plane_index, coordinate_index in enumerate(coordinate_indices):
        index = (
            np.searchsorted(
                edges[plane_index], selected_positions[:, coordinate_index], side="right"
            )
            - 1
        )
        bin_indices.append(np.clip(index, 0, bins[plane_index] - 1))
    flat = np.ravel_multi_index(tuple(bin_indices), bins)
    return edges, selected_ids, flat


def _bin_metadata(
    edges: tuple[np.ndarray, np.ndarray],
    axis_labels: tuple[str, str],
    bins: tuple[int, int],
    atom_bins: np.ndarray,
) -> pd.DataFrame:
    flat = np.arange(int(np.prod(bins)), dtype=int)
    plane_indices = np.unravel_index(flat, bins)
    values: dict[str, np.ndarray] = {"bin_number": flat + 1}
    for axis in "xyz":
        values[f"bin_{axis}"] = np.full(flat.size, -1, dtype=int)
        values[f"{axis}_min"] = np.full(flat.size, np.nan)
        values[f"{axis}_max"] = np.full(flat.size, np.nan)
        values[f"{axis}_center"] = np.full(flat.size, np.nan)
    for plane_index, axis in enumerate(axis_labels):
        index = plane_indices[plane_index]
        axis_edges = edges[plane_index]
        values[f"bin_{axis}"] = index
        values[f"{axis}_min"] = axis_edges[index]
        values[f"{axis}_max"] = axis_edges[index + 1]
        values[f"{axis}_center"] = (axis_edges[index] + axis_edges[index + 1]) / 2.0
    values["atom_count"] = np.bincount(atom_bins, minlength=flat.size)
    return pd.DataFrame(values)


def _set_charge_aggregates(
    frame_table: pd.DataFrame,
    fixed_bins: np.ndarray,
    valid: np.ndarray,
    frame_charges: np.ndarray,
    baseline: np.ndarray,
    number_of_bins: int,
    *,
    include_averages: bool,
) -> None:
    from reaxkit.core.runtime.reducers import CountSumReducer
    charge = CountSumReducer(number_of_bins)
    delta = CountSumReducer(number_of_bins)
    charge.add(fixed_bins[valid], frame_charges[valid])
    delta.add(fixed_bins[valid], frame_charges[valid] - baseline[valid])
    counts, charge_sums, _ = charge.finalize()
    _, delta_sums, _ = delta.finalize()
    frame_table["valid_charge_count"] = counts
    frame_table["charge"] = charge_sums
    frame_table["delta_charge"] = delta_sums
    if include_averages:
        frame_table["average_charge"] = np.divide(
            charge_sums,
            counts,
            out=np.full(number_of_bins, np.nan),
            where=counts > 0,
        )
        frame_table["average_delta_charge"] = np.divide(
            delta_sums,
            counts,
            out=np.full(number_of_bins, np.nan),
            where=counts > 0,
        )


def _result_columns(request: BinnedDynamicChargeRequest) -> list[str]:
    return [*BIN_COLUMNS, *AVERAGE_COLUMNS] if request.average else BIN_COLUMNS


def calculate_binned_dynamic_charges(
    data: ElectrostaticsData,
    request: BinnedDynamicChargeRequest,
) -> BinnedDynamicChargeResult:
    """Sum charge and frame-zero charge change after projection onto a plane."""

    plane, axis_labels, bins = _plane_configuration(request)
    positions = np.asarray(data.trajectory.positions, dtype=float)
    charges = np.asarray(data.charges.charges, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("Trajectory positions must have shape (n_frames, n_atoms, 3).")
    if charges.ndim != 2:
        raise ValueError("Charges must have shape (n_frames, n_atoms).")
    if positions.shape[0] != charges.shape[0]:
        raise ValueError("Trajectory positions and charges must have the same frame count.")

    trajectory_ids = np.asarray(data.trajectory.atom_ids, dtype=int)
    charge_ids = np.asarray(
        data.charges.simulation.atom_ids
        if data.charges.simulation is not None
        else data.trajectory.atom_ids,
        dtype=int,
    )
    if trajectory_ids.size != positions.shape[1] or charge_ids.size != charges.shape[1]:
        raise ValueError("Atom-id dimensions must match positions and charges.")
    if np.unique(trajectory_ids).size != trajectory_ids.size or np.unique(charge_ids).size != charge_ids.size:
        raise ValueError("Atom ids must be unique.")
    charge_index = {int(atom_id): index for index, atom_id in enumerate(charge_ids)}
    missing = [int(atom_id) for atom_id in trajectory_ids if int(atom_id) not in charge_index]
    if missing:
        raise ValueError(f"Trajectory atom id(s) missing from charges: {missing}.")

    edges, fixed_atom_ids, fixed_bins = _fixed_grid(
        positions[0], trajectory_ids, axis_labels, bins
    )
    charge_columns = np.asarray(
        [charge_index[int(atom_id)] for atom_id in fixed_atom_ids], dtype=int
    )
    baseline = charges[0, charge_columns]
    baseline_valid = np.isfinite(baseline)
    fixed_atom_ids = fixed_atom_ids[baseline_valid]
    fixed_bins = fixed_bins[baseline_valid]
    charge_columns = charge_columns[baseline_valid]
    baseline = baseline[baseline_valid]
    if baseline.size == 0:
        raise ValueError("Frame 0 must contain at least one finite atomic charge.")

    frames = _selected_frames(charges.shape[0], request)
    number_of_bins = int(np.prod(bins))
    metadata = _bin_metadata(edges, axis_labels, bins, fixed_bins)
    all_iterations = (
        np.asarray(data.charges.iterations, dtype=int)
        if data.charges.iterations is not None
        else np.asarray(data.trajectory.iterations, dtype=int)
        if data.trajectory.iterations is not None
        else np.arange(charges.shape[0], dtype=int)
    )
    if all_iterations.shape != (charges.shape[0],):
        raise ValueError("Iteration count must match the number of frames.")

    rows: list[pd.DataFrame] = []
    for frame in frames:
        frame_charges = charges[frame, charge_columns]
        valid = np.isfinite(frame_charges)
        frame_table = metadata.copy()
        frame_table.insert(0, "iteration", int(all_iterations[frame]))
        frame_table.insert(0, "frame", int(frame))
        _set_charge_aggregates(
            frame_table,
            fixed_bins,
            valid,
            frame_charges,
            baseline,
            number_of_bins,
            include_averages=bool(request.average),
        )
        rows.append(frame_table)

    table = (
        pd.concat(rows, ignore_index=True)[_result_columns(request)]
        if rows
        else pd.DataFrame(columns=_result_columns(request))
    )
    time_values = None
    simulation = data.trajectory.simulation or data.charges.simulation
    if simulation is not None and simulation.time is not None:
        all_times = np.asarray(simulation.time, dtype=float)
        if all_times.shape != (charges.shape[0],):
            raise ValueError("Simulation time count must match the number of frames.")
        time_values = all_times[frames]

    return BinnedDynamicChargeResult(
        table=table,
        request=request,
        frame_indices=frames,
        iterations=all_iterations[frames],
        plane=plane,
        axis_labels=axis_labels,
        bin_edges=edges,
        atom_bin_numbers={
            int(atom_id): int(bin_index) + 1
            for atom_id, bin_index in zip(fixed_atom_ids, fixed_bins, strict=False)
        },
        time_values=time_values,
    )


def _global_color_limits(values: pd.Series, *, symmetric: bool) -> tuple[float, float]:
    finite = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-1.0, 1.0)
    if symmetric:
        bound = float(np.max(np.abs(finite)))
        return (-bound, bound) if bound > 0.0 else (-1.0, 1.0)
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if lower == upper:
        padding = max(abs(lower) * 0.05, 0.05)
        return lower - padding, upper + padding
    return lower, upper


def generate_binned_charge_heatmaps(
    result: BinnedDynamicChargeResult,
    output_root: Path,
    *,
    dpi: int = 180,
    progress: bool = True,
) -> list[Path]:
    """Write globally color-scaled 2-D heatmaps on the selected plane."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Heatmap generation requires matplotlib; install reaxkit[plot].") from exc

    output_root = Path(output_root)
    quantities = (
        ("average_charge", "average_delta_charge")
        if result.request.average
        else ("charge", "delta_charge")
    )
    occupied_rows = result.table["valid_charge_count"] > 0
    limits = {
        quantities[0]: _global_color_limits(
            result.table.loc[occupied_rows, quantities[0]], symmetric=False
        ),
        quantities[1]: _global_color_limits(
            result.table.loc[occupied_rows, quantities[1]], symmetric=True
        ),
    }
    bins_by_axis = {
        "x": result.request.bins_x,
        "y": result.request.bins_y,
        "z": result.request.bins_z,
    }
    shape = tuple(int(bins_by_axis[axis]) for axis in result.axis_labels)
    written: list[Path] = []
    progress_bar = None
    if progress:
        from tqdm.auto import tqdm

        progress_bar = tqdm(
            total=2 * len(result.frame_indices),
            desc=f"plots: generating {result.plane} charge heatmaps",
            unit="plot",
            dynamic_ncols=True,
        )

    try:
        for quantity in quantities:
            destination_dir = output_root / "heatmaps" / quantity
            destination_dir.mkdir(parents=True, exist_ok=True)
            norm = Normalize(*limits[quantity])
            for frame in result.frame_indices:
                frame_table = result.table[result.table["frame"] == int(frame)]
                values = frame_table[quantity].to_numpy(dtype=float).reshape(shape).T
                figure, axis = plt.subplots(figsize=(7.2, 5.8))
                image = axis.pcolormesh(
                    result.bin_edges[0],
                    result.bin_edges[1],
                    values,
                    cmap=HEATMAP_CMAP,
                    norm=norm,
                    shading="flat",
                )
                axis.set_xlim(result.bin_edges[0][0], result.bin_edges[0][-1])
                axis.set_ylim(result.bin_edges[1][0], result.bin_edges[1][-1])
                axis.set_xlabel(result.axis_labels[0])
                axis.set_ylabel(result.axis_labels[1])
                axis.set_aspect("equal", adjustable="box")
                iteration = int(frame_table["iteration"].iloc[0])
                summed_axis = ({"x", "y", "z"} - set(result.axis_labels)).pop()
                is_delta = quantity in {"delta_charge", "average_delta_charge"}
                is_average = quantity.startswith("average_")
                base_label = "delta charge from frame 0" if is_delta else "charge"
                label = f"average {base_label}" if is_average else base_label
                axis.set_title(
                    f"Binned {label} on {result.plane} (summed over {summed_axis})\n"
                    f"frame {int(frame)} | iteration {iteration}"
                )
                colorbar = figure.colorbar(image, ax=axis)
                colorbar.set_label(label if is_average else f"summed {label}")
                figure.tight_layout()
                destination = destination_dir / f"frame_{int(frame):06d}.png"
                figure.savefig(destination, dpi=int(dpi), bbox_inches="tight")
                plt.close(figure)
                written.append(destination)
                if progress_bar is not None:
                    progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()
    return written


@register_task("get_binned_dynamic_charges", label="Binned Dynamic Charges")
class BinnedDynamicChargeTask(AnalysisTask):
    """Aggregate atomic charges on a projected grid fixed at frame zero."""

    required_data = ElectrostaticsData
    execution_capabilities = TaskCapabilities(shape=ExecutionShape.REFERENCE_FRAME_MAP,
        supports_selective_frames=True, reference_frames=(0,), estimated_frame_bytes=8 * 1024 * 1024)
    VERSION = "2"

    @staticmethod
    def required_data_fields_for(
        _request: BinnedDynamicChargeRequest,
        _args: dict,
    ) -> tuple[str, ...]:
        return ("trajectory", "charges")

    @staticmethod
    def recommended_presentations(
        _result: BinnedDynamicChargeResult,
        _payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        return [PresentationSpec(renderer="table", label="Binned charges", view_type="table")]

    def run(
        self,
        data: ElectrostaticsData,
        request: BinnedDynamicChargeRequest,
        reporter=None,
    ) -> BinnedDynamicChargeResult:
        _ = reporter
        return calculate_binned_dynamic_charges(data, request)

    def run_stream(
        self,
        frames,
        request: BinnedDynamicChargeRequest,
        reporter=None,
    ) -> BinnedDynamicChargeResult:
        """Aggregate charge-only frames without materializing ``fort.7``."""

        plane, axis_labels, bins = _plane_configuration(request)
        requested = (
            None
            if request.selected_frames is None
            else [int(value) for value in request.selected_frames][:: int(request.every)]
        )
        requested_set = set(requested or ())
        number_of_bins = int(np.prod(bins))
        metadata = None
        fixed_atom_ids = None
        fixed_bins = None
        baseline = None
        edges = None
        rows: list[pd.DataFrame] = []
        output_frames: list[int] = []
        output_iterations: list[int] = []
        output_times: list[float] = []
        have_all_times = True
        seen_frames: set[int] = set()
        processed = 0

        for stream_index, data in enumerate(frames):
            processed += 1
            source_values = data.trajectory.source_frame_indices
            source_frame = (
                int(np.asarray(source_values).reshape(-1)[0])
                if source_values is not None
                else stream_index
            )
            seen_frames.add(source_frame)
            positions = np.asarray(data.trajectory.positions, dtype=float)
            charges = np.asarray(data.charges.charges, dtype=float)
            if positions.shape[0] != 1 or charges.shape[0] != 1:
                raise ValueError("Streamed electrostatics data must contain one frame.")
            trajectory_ids = np.asarray(data.trajectory.atom_ids, dtype=int)
            charge_ids = np.asarray(
                data.charges.simulation.atom_ids
                if data.charges.simulation is not None
                else data.trajectory.atom_ids,
                dtype=int,
            )
            charge_index = {int(atom_id): index for index, atom_id in enumerate(charge_ids)}

            if baseline is None:
                if source_frame != 0:
                    raise ValueError("Streaming must begin at frame 0 to define fixed bins.")
                edges, fixed_atom_ids, fixed_bins = _fixed_grid(
                    positions[0], trajectory_ids, axis_labels, bins
                )
                missing = [
                    int(atom_id)
                    for atom_id in fixed_atom_ids
                    if int(atom_id) not in charge_index
                ]
                if missing:
                    raise ValueError(f"Trajectory atom id(s) missing from charges: {missing}.")
                baseline_columns = np.asarray(
                    [charge_index[int(atom_id)] for atom_id in fixed_atom_ids], dtype=int
                )
                baseline_values = charges[0, baseline_columns]
                valid_baseline = np.isfinite(baseline_values)
                fixed_atom_ids = fixed_atom_ids[valid_baseline]
                fixed_bins = fixed_bins[valid_baseline]
                baseline = baseline_values[valid_baseline]
                if baseline.size == 0:
                    raise ValueError("Frame 0 must contain at least one finite atomic charge.")
                metadata = _bin_metadata(edges, axis_labels, bins, fixed_bins)

            keep = (
                source_frame in requested_set
                if requested is not None
                else source_frame % int(request.every) == 0
            )
            if keep:
                missing = [
                    int(atom_id)
                    for atom_id in fixed_atom_ids
                    if int(atom_id) not in charge_index
                ]
                if missing:
                    raise ValueError(f"Frame {source_frame} is missing atom id(s): {missing}.")
                columns = np.asarray(
                    [charge_index[int(atom_id)] for atom_id in fixed_atom_ids], dtype=int
                )
                frame_charges = charges[0, columns]
                valid = np.isfinite(frame_charges)
                iteration = (
                    int(np.asarray(data.charges.iterations).reshape(-1)[0])
                    if data.charges.iterations is not None
                    else source_frame
                )
                frame_table = metadata.copy()
                frame_table.insert(0, "iteration", iteration)
                frame_table.insert(0, "frame", source_frame)
                _set_charge_aggregates(
                    frame_table,
                    fixed_bins,
                    valid,
                    frame_charges,
                    baseline,
                    number_of_bins,
                    include_averages=bool(request.average),
                )
                rows.append(frame_table)
                output_frames.append(source_frame)
                output_iterations.append(iteration)
                simulation = data.trajectory.simulation
                if simulation is None or simulation.time is None:
                    have_all_times = False
                else:
                    output_times.append(float(np.asarray(simulation.time).reshape(-1)[0]))
            if callable(reporter):
                reporter(
                    "stream",
                    processed,
                    int(request._expected_frames or 0),
                    "Binning projected charge frames",
                )

        if baseline is None or edges is None or fixed_atom_ids is None or fixed_bins is None:
            raise ValueError("No frame-zero charge data was streamed.")
        if requested is not None:
            missing_frames = [frame for frame in requested if frame not in seen_frames]
            if missing_frames:
                raise ValueError(f"Requested frame(s) not found: {missing_frames}.")
        if callable(reporter):
            reporter("stream", processed, processed, "Finished binned charge streaming")
        table = (
            pd.concat(rows, ignore_index=True)[_result_columns(request)]
            if rows
            else pd.DataFrame(columns=_result_columns(request))
        )
        return BinnedDynamicChargeResult(
            table=table,
            request=request,
            frame_indices=np.asarray(output_frames, dtype=int),
            iterations=np.asarray(output_iterations, dtype=int),
            plane=plane,
            axis_labels=axis_labels,
            bin_edges=edges,
            atom_bin_numbers={
                int(atom_id): int(bin_index) + 1
                for atom_id, bin_index in zip(fixed_atom_ids, fixed_bins, strict=False)
            },
            time_values=(
                np.asarray(output_times, dtype=float) if have_all_times else None
            ),
        )


__all__ = [
    "AVERAGE_COLUMNS", "BIN_COLUMNS", "HEATMAP_CMAP",
    "BinnedDynamicChargeRequest", "BinnedDynamicChargeResult",
    "BinnedDynamicChargeTask", "calculate_binned_dynamic_charges",
    "generate_binned_charge_heatmaps",
]
