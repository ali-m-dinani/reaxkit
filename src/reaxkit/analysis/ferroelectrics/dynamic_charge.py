"""Analyze per-atom charge changes relative to simulation frame zero.

The analyzer consumes canonical :class:`~reaxkit.domain.data_models.ChargeData`
and produces a long-form charge table plus per-atom descriptive statistics.
File loading, CSV persistence, and plotting remain workflow responsibilities.
"""

from __future__ import annotations

import csv
import math
import tempfile
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ChargeData
from reaxkit.presentation.specs import PresentationSpec

DETAIL_COLUMNS = ["frame", "atom_number", "atom_type", "charge", "delta_charge"]
SUMMARY_COLUMNS = [
    "atom_number",
    "atom_type",
    "mean(charge)",
    "median(charge)",
    "min(charge)",
    "max(charge)",
    "std(charge)",
    "range(charge)",
    "mean(delta_charge)",
    "median(delta_charge)",
    "min(delta_charge)",
    "max(delta_charge)",
    "std(delta_charge)",
    "range(delta_charge)",
]
FRAME_SUMMARY_COLUMNS = ["frame", *SUMMARY_COLUMNS[2:]]
FRAME_ATOM_TYPE_SUMMARY_COLUMNS = ["frame", "atom_type", *SUMMARY_COLUMNS[2:]]


@dataclass
class DynamicChargeChangeRequest(BaseRequest):
    """Configuration for charge changes measured from frame zero."""

    atom_numbers: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={"label": "Atom numbers", "help": "Optional atom numbers; omit for all atoms."},
    )
    selected_frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={"label": "Frames", "help": "Frames to report; frame zero is always the baseline."},
    )
    every: int = dc_field(
        default=1,
        metadata={"label": "Every", "help": "Keep every Nth selected frame.", "min": 1},
    )
    _detail_csv_path: Optional[str] = None
    _matrix_path: Optional[str] = None
    _expected_frames: Optional[int] = None
    _retain_matrix: bool = False


@dataclass
class DynamicChargeChangeResult(BaseResult):
    """Detailed charge changes, per-atom summary statistics, and plot axes."""

    charges: pd.DataFrame
    summary: pd.DataFrame
    summary_per_frame_for_all_atoms: pd.DataFrame
    summary_per_frame_per_atom_type: pd.DataFrame
    request: DynamicChargeChangeRequest
    frame_indices: np.ndarray
    iterations: np.ndarray
    time_values: Optional[np.ndarray] = None
    detail_csv_path: Optional[str] = None
    charge_matrix_path: Optional[str] = None
    charge_matrix_shape: Optional[tuple[int, int]] = None
    atom_numbers: np.ndarray = dc_field(default_factory=lambda: np.empty(0, dtype=int))
    atom_types: list[str] = dc_field(default_factory=list)
    matrix_columns: np.ndarray = dc_field(default_factory=lambda: np.empty(0, dtype=int))
    baseline_charges: np.ndarray = dc_field(default_factory=lambda: np.empty(0, dtype=float))
    detail_row_count: int = 0

    @property
    def table(self) -> pd.DataFrame:
        """Return the compact summary for terminal and UI presentation."""

        return self.summary

    @property
    def csv_tables(self) -> dict[str, pd.DataFrame]:
        """Persist only the summary through pandas; detail may already be chunk-written."""

        return {
            "summary_per_atom": self.summary,
            "summary_per_frame_for_all_atoms": self.summary_per_frame_for_all_atoms,
            "summary_per_frame_per_atom_type": self.summary_per_frame_per_atom_type,
        }

    @property
    def prewritten_csvs(self) -> list[str]:
        """Return streamed CSV artifacts already written by the analyzer."""

        return [self.detail_csv_path] if self.detail_csv_path else []


@dataclass
class _RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, value: float) -> None:
        if not np.isfinite(value):
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def std(self) -> float:
        return math.sqrt(self.m2 / (self.count - 1)) if self.count > 1 else float("nan")


def _selected_frames(
        n_frames: int,
        requested: Optional[Sequence[int]],
        every: int,
) -> np.ndarray:
    if n_frames < 1:
        raise ValueError("ChargeData must contain at least one frame so frame 0 can be the baseline.")
    if every < 1:
        raise ValueError("every must be at least 1.")
    frames = np.arange(n_frames, dtype=int) if requested is None else np.asarray(requested, dtype=int)
    if frames.ndim != 1:
        raise ValueError("selected_frames must be a one-dimensional sequence.")
    if np.any(frames < 0) or np.any(frames >= n_frames):
        raise ValueError(f"selected_frames must be between 0 and {n_frames - 1}.")
    return frames[::every]


def _atom_identity(data: ChargeData, n_atoms: int) -> tuple[list[int], list[object]]:
    simulation = data.simulation
    atom_numbers = (
        [int(value) for value in simulation.atom_ids]
        if simulation is not None
        else list(range(1, n_atoms + 1))
    )
    if len(atom_numbers) != n_atoms:
        raise ValueError("ChargeData atom_ids length must match the charge atom dimension.")

    if simulation is not None and simulation.elements is not None:
        atom_types: list[object] = [str(value) for value in simulation.elements]
    elif simulation is not None and simulation.atom_type_nums is not None:
        type_numbers = np.asarray(simulation.atom_type_nums)
        atom_types = [
            int(value) if np.isfinite(value) else ""
            for value in type_numbers[0]
        ]
    else:
        atom_types = [""] * n_atoms
    if len(atom_types) != n_atoms:
        raise ValueError("ChargeData atom types length must match the charge atom dimension.")
    return atom_numbers, atom_types


def _summary_table(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    grouped = detail.groupby(["atom_number", "atom_type"], sort=False, dropna=False)
    summary = grouped.agg(
        **{
            "mean(charge)": ("charge", "mean"),
            "median(charge)": ("charge", "median"),
            "min(charge)": ("charge", "min"),
            "max(charge)": ("charge", "max"),
            "std(charge)": ("charge", "std"),
            "mean(delta_charge)": ("delta_charge", "mean"),
            "median(delta_charge)": ("delta_charge", "median"),
            "min(delta_charge)": ("delta_charge", "min"),
            "max(delta_charge)": ("delta_charge", "max"),
            "std(delta_charge)": ("delta_charge", "std"),
        }
    ).reset_index()
    summary["range(charge)"] = summary["max(charge)"] - summary["min(charge)"]
    summary["range(delta_charge)"] = (
            summary["max(delta_charge)"] - summary["min(delta_charge)"]
    )
    return summary[SUMMARY_COLUMNS]


def _grouped_summary_table(
        detail: pd.DataFrame,
        group_columns: list[str],
        output_columns: list[str],
) -> pd.DataFrame:
    """Calculate charge and delta-charge statistics for compact grouped output."""

    if detail.empty:
        return pd.DataFrame(columns=output_columns)
    grouped = detail.groupby(group_columns, sort=False, dropna=False)
    summary = grouped.agg(
        **{
            "mean(charge)": ("charge", "mean"),
            "median(charge)": ("charge", "median"),
            "min(charge)": ("charge", "min"),
            "max(charge)": ("charge", "max"),
            "std(charge)": ("charge", "std"),
            "mean(delta_charge)": ("delta_charge", "mean"),
            "median(delta_charge)": ("delta_charge", "median"),
            "min(delta_charge)": ("delta_charge", "min"),
            "max(delta_charge)": ("delta_charge", "max"),
            "std(delta_charge)": ("delta_charge", "std"),
        }
    ).reset_index()
    summary["range(charge)"] = summary["max(charge)"] - summary["min(charge)"]
    summary["range(delta_charge)"] = (
            summary["max(delta_charge)"] - summary["min(delta_charge)"]
    )
    return summary[output_columns]


def _statistics(values: Sequence[float], prefix: str) -> dict[str, float]:
    """Return pandas-compatible descriptive statistics for one in-memory group."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            f"mean({prefix})": float("nan"),
            f"median({prefix})": float("nan"),
            f"min({prefix})": float("nan"),
            f"max({prefix})": float("nan"),
            f"std({prefix})": float("nan"),
            f"range({prefix})": float("nan"),
        }
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    return {
        f"mean({prefix})": float(np.mean(array)),
        f"median({prefix})": float(np.median(array)),
        f"min({prefix})": minimum,
        f"max({prefix})": maximum,
        f"std({prefix})": float(np.std(array, ddof=1)) if array.size > 1 else float("nan"),
        f"range({prefix})": maximum - minimum,
    }


def _frame_summary_row(
        frame: int,
        charges: Sequence[float],
        delta_charges: Sequence[float],
        *,
        atom_type: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {"frame": int(frame)}
    if atom_type is not None:
        row["atom_type"] = atom_type
    row.update(_statistics(charges, "charge"))
    row.update(_statistics(delta_charges, "delta_charge"))
    return row


def calculate_dynamic_charge_changes(
        data: ChargeData,
        request: DynamicChargeChangeRequest,
) -> DynamicChargeChangeResult:
    """Calculate charges and ``charge(frame) - charge(frame 0)`` for each atom."""

    charges = np.asarray(data.charges, dtype=float)
    if charges.ndim != 2:
        raise ValueError("ChargeData.charges must have shape (n_frames, n_atoms).")
    n_frames, n_atoms = charges.shape
    frames = _selected_frames(n_frames, request.selected_frames, int(request.every))
    atom_numbers, atom_types = _atom_identity(data, n_atoms)
    atom_to_index = {number: index for index, number in enumerate(atom_numbers)}
    selected_atoms = atom_numbers if request.atom_numbers is None else [int(v) for v in request.atom_numbers]
    missing = [number for number in selected_atoms if number not in atom_to_index]
    if missing:
        raise ValueError(f"Atom number(s) not found in ChargeData: {missing}.")

    baseline = charges[0]
    rows: list[dict[str, object]] = []
    for frame in frames:
        for atom_number in selected_atoms:
            atom_index = atom_to_index[atom_number]
            charge = float(charges[frame, atom_index])
            if not np.isfinite(charge):
                continue
            rows.append(
                {
                    "frame": int(frame),
                    "atom_number": atom_number,
                    "atom_type": atom_types[atom_index],
                    "charge": charge,
                    "delta_charge": charge - float(baseline[atom_index]),
                }
            )
    detail = pd.DataFrame(rows, columns=DETAIL_COLUMNS)

    iterations = (
        np.asarray(data.iterations, dtype=int)
        if data.iterations is not None
        else np.arange(n_frames, dtype=int)
    )
    if iterations.shape != (n_frames,):
        raise ValueError("ChargeData.iterations length must match the charge frame dimension.")
    time_values = None
    if data.simulation is not None and data.simulation.time is not None:
        all_times = np.asarray(data.simulation.time, dtype=float)
        if all_times.shape != (n_frames,):
            raise ValueError("ChargeData.simulation.time length must match the charge frame dimension.")
        time_values = all_times[frames]

    return DynamicChargeChangeResult(
        charges=detail,
        summary=_summary_table(detail),
        summary_per_frame_for_all_atoms=_grouped_summary_table(
            detail,
            ["frame"],
            FRAME_SUMMARY_COLUMNS,
        ),
        summary_per_frame_per_atom_type=_grouped_summary_table(
            detail,
            ["frame", "atom_type"],
            FRAME_ATOM_TYPE_SUMMARY_COLUMNS,
        ),
        request=request,
        frame_indices=frames,
        iterations=iterations[frames],
        time_values=time_values,
    )


@register_task("get_dynamic_charge_changes", label="Dynamic Charge Changes")
class DynamicChargeChangeTask(AnalysisTask):
    """Compute per-atom dynamic charge changes from canonical charge data."""

    required_data = ChargeData
    VERSION = "3"

    @staticmethod
    def recommended_presentations(
            _result: DynamicChargeChangeResult,
            _payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        return [PresentationSpec(renderer="table", label="Charges", view_type="table")]

    def run(
            self,
            data: ChargeData,
            request: DynamicChargeChangeRequest,
            reporter=None,
    ) -> DynamicChargeChangeResult:
        _ = reporter
        return calculate_dynamic_charge_changes(data, request)

    def run_stream(
            self,
            frames,
            request: DynamicChargeChangeRequest,
            reporter=None,
    ) -> DynamicChargeChangeResult:
        """Chunk-write details and aggregate statistics with atom-scale memory."""

        if int(request.every) < 1:
            raise ValueError("every must be at least 1.")
        requested = (
            None
            if request.selected_frames is None
            else [int(value) for value in request.selected_frames][:: int(request.every)]
        )
        requested_set = set(requested or [])
        selected_atom_numbers = (
            None if request.atom_numbers is None else [int(value) for value in request.atom_numbers]
        )
        if request._detail_csv_path:
            output_dir = Path(request._detail_csv_path).parent
        elif request._matrix_path:
            output_dir = Path(request._matrix_path).parent
        else:
            output_dir = Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        detail_path = Path(request._detail_csv_path) if request._detail_csv_path else None
        if request._matrix_path:
            matrix_path = Path(request._matrix_path)
        else:
            matrix_file = tempfile.NamedTemporaryFile(
                prefix=".dynamic_charge_matrix_",
                suffix=".dat",
                dir=output_dir,
                delete=False,
            )
            matrix_path = Path(matrix_file.name)
            matrix_file.close()
        spool = tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=".dynamic_charge_frames_",
            suffix=".npyseq",
            dir=output_dir,
            delete=False,
        )
        spool_path = Path(spool.name)
        detail_handle = (
            detail_path.open("w", encoding="utf-8", newline="")
            if detail_path is not None
            else None
        )
        writer = csv.writer(detail_handle) if detail_handle is not None else None
        if writer is not None:
            writer.writerow(DETAIL_COLUMNS)

        baseline: dict[int, float] | None = None
        charge_stats: dict[int, _RunningStats] = {}
        delta_stats: dict[int, _RunningStats] = {}
        atom_type_by_number: dict[int, str] = {}
        preview_rows: list[dict[str, object]] = []
        output_frames: list[int] = []
        output_iterations: list[int] = []
        output_times: list[float] = []
        have_all_times = True
        seen_frames: set[int] = set()
        seen_atom_numbers: set[int] = set()
        matrix_column_by_atom: dict[int, int] = {}
        detail_row_count = 0
        processed_frames = 0
        max_atoms = 0
        frame_summary_rows: list[dict[str, object]] = []
        frame_atom_type_summary_rows: list[dict[str, object]] = []

        stream_succeeded = False
        try:
            for stream_index, data in enumerate(frames):
                processed_frames += 1
                metadata = data.metadata or {}
                source_values = metadata.get("source_frame_indices")
                source_frame = (
                    int(np.asarray(source_values).reshape(-1)[0])
                    if source_values is not None
                    else stream_index
                )
                seen_frames.add(source_frame)
                charges = np.asarray(data.charges, dtype=float)
                if charges.shape[0] != 1:
                    raise ValueError("Streamed ChargeData must contain exactly one frame.")
                atom_numbers, atom_types = _atom_identity(data, charges.shape[1])
                atom_to_index = {number: index for index, number in enumerate(atom_numbers)}
                present_atom_numbers = [
                    number
                    for number, index in atom_to_index.items()
                    if np.isfinite(charges[0, index])
                ]
                for atom_number in present_atom_numbers:
                    if atom_number not in matrix_column_by_atom:
                        matrix_column_by_atom[atom_number] = len(matrix_column_by_atom)
                seen_atom_numbers.update(present_atom_numbers)
                for atom_number in present_atom_numbers:
                    atom_type_by_number[atom_number] = str(atom_types[atom_to_index[atom_number]])
                if baseline is None:
                    if source_frame != 0:
                        raise ValueError("Dynamic charge streaming must begin at frame 0.")
                    baseline = {
                        number: float(charges[0, index])
                        for number, index in atom_to_index.items()
                        if np.isfinite(charges[0, index])
                    }

                keep = (
                    source_frame in requested_set
                    if requested is not None
                    else source_frame % int(request.every) == 0
                )
                if keep:
                    chosen_atoms = (
                        present_atom_numbers
                        if selected_atom_numbers is None
                        else [number for number in selected_atom_numbers if number in present_atom_numbers]
                    )
                    iteration = (
                        int(np.asarray(data.iterations).reshape(-1)[0])
                        if data.iterations is not None
                        else source_frame
                    )
                    frame_values = np.full(len(matrix_column_by_atom), np.nan, dtype=float)
                    csv_rows = []
                    frame_charges: list[float] = []
                    frame_deltas: list[float] = []
                    charges_by_type: dict[str, list[float]] = {}
                    deltas_by_type: dict[str, list[float]] = {}
                    frame_row_count = 0
                    for atom_number in chosen_atoms:
                        atom_index = atom_to_index[atom_number]
                        charge = float(charges[0, atom_index])
                        delta = charge - baseline.get(atom_number, float("nan"))
                        atom_type = str(atom_types[atom_index])
                        frame_values[matrix_column_by_atom[atom_number]] = charge
                        charge_stats.setdefault(atom_number, _RunningStats()).update(charge)
                        delta_stats.setdefault(atom_number, _RunningStats()).update(delta)
                        frame_charges.append(charge)
                        frame_deltas.append(delta)
                        charges_by_type.setdefault(atom_type, []).append(charge)
                        deltas_by_type.setdefault(atom_type, []).append(delta)
                        if len(preview_rows) < 20:
                            preview_rows.append(
                                {
                                    "frame": source_frame,
                                    "atom_number": atom_number,
                                    "atom_type": atom_type,
                                    "charge": charge,
                                    "delta_charge": delta,
                                }
                            )
                        if writer is not None:
                            csv_rows.append(
                                (source_frame, atom_number, atom_type, charge, delta)
                            )
                        frame_row_count += 1
                    if writer is not None:
                        writer.writerows(csv_rows)
                    detail_row_count += frame_row_count
                    if frame_charges:
                        frame_summary_rows.append(
                            _frame_summary_row(source_frame, frame_charges, frame_deltas)
                        )
                    for atom_type, type_charges in charges_by_type.items():
                        frame_atom_type_summary_rows.append(
                            _frame_summary_row(
                                source_frame,
                                type_charges,
                                deltas_by_type[atom_type],
                                atom_type=atom_type,
                            )
                        )
                    np.save(spool, frame_values, allow_pickle=False)
                    max_atoms = max(max_atoms, frame_values.size)
                    output_frames.append(source_frame)
                    output_iterations.append(iteration)
                    time_values = data.simulation.time if data.simulation is not None else None
                    if time_values is None:
                        have_all_times = False
                    else:
                        output_times.append(float(np.asarray(time_values).reshape(-1)[0]))
                if callable(reporter):
                    reporter(
                        "stream",
                        processed_frames,
                        int(request._expected_frames or 0),
                        "Reading charge and atom-identity frames",
                    )
            stream_succeeded = True
        finally:
            spool.flush()
            spool.close()
            if detail_handle is not None:
                detail_handle.close()
            if not stream_succeeded:
                spool_path.unlink(missing_ok=True)
                matrix_path.unlink(missing_ok=True)

        if baseline is None:
            spool_path.unlink(missing_ok=True)
            raise ValueError("ChargeData must contain at least one frame.")
        if requested is not None:
            missing_frames = [frame for frame in requested if frame not in seen_frames]
            if missing_frames:
                spool_path.unlink(missing_ok=True)
                raise ValueError(f"Requested frame(s) not found in ChargeData: {missing_frames}.")
        if selected_atom_numbers is not None:
            missing_atoms = [number for number in selected_atom_numbers if number not in seen_atom_numbers]
            if missing_atoms:
                spool_path.unlink(missing_ok=True)
                raise ValueError(f"Atom number(s) not found in ChargeData: {missing_atoms}.")
        if callable(reporter):
            reporter("stream", processed_frames, processed_frames, "Finished reading charge frames")

        matrix = np.memmap(
            matrix_path,
            mode="w+",
            dtype=np.float64,
            shape=(len(output_frames), max_atoms),
        )
        matrix[:] = np.nan
        with spool_path.open("rb") as handle:
            for frame_index in range(len(output_frames)):
                values = np.load(handle, allow_pickle=False)
                matrix[frame_index, : values.size] = values
        matrix.flush()
        spool_path.unlink(missing_ok=True)

        summary_rows: list[dict[str, object]] = []
        reported_atoms = sorted(charge_stats)
        for atom_number in reported_atoms:
            matrix_column = matrix_column_by_atom[atom_number]
            # Copy the column so Windows can release the underlying memmap file.
            charge_values = np.array(matrix[:, matrix_column], dtype=float, copy=True)
            reference = baseline.get(atom_number, float("nan"))
            delta_values = charge_values - reference
            charge = charge_stats[atom_number]
            delta = delta_stats[atom_number]
            summary_rows.append(
                {
                    "atom_number": atom_number,
                    "atom_type": atom_type_by_number.get(atom_number, ""),
                    "mean(charge)": charge.mean,
                    "median(charge)": float(np.nanmedian(charge_values)),
                    "min(charge)": charge.minimum,
                    "max(charge)": charge.maximum,
                    "std(charge)": charge.std(),
                    "range(charge)": charge.maximum - charge.minimum,
                    "mean(delta_charge)": delta.mean if delta.count else float("nan"),
                    "median(delta_charge)": (
                        float(np.nanmedian(delta_values)) if delta.count else float("nan")
                    ),
                    "min(delta_charge)": delta.minimum if delta.count else float("nan"),
                    "max(delta_charge)": delta.maximum if delta.count else float("nan"),
                    "std(delta_charge)": delta.std(),
                    "range(delta_charge)": (
                        delta.maximum - delta.minimum if delta.count else float("nan")
                    ),
                }
            )
        summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
        del matrix
        retained_matrix_path = str(matrix_path) if request._retain_matrix else None
        if not request._retain_matrix:
            matrix_path.unlink(missing_ok=True)
        baseline_array = np.asarray(
            [baseline.get(atom_number, float("nan")) for atom_number in reported_atoms],
            dtype=float,
        )
        return DynamicChargeChangeResult(
            charges=pd.DataFrame(preview_rows, columns=DETAIL_COLUMNS),
            summary=summary,
            summary_per_frame_for_all_atoms=pd.DataFrame(
                frame_summary_rows,
                columns=FRAME_SUMMARY_COLUMNS,
            ),
            summary_per_frame_per_atom_type=pd.DataFrame(
                frame_atom_type_summary_rows,
                columns=FRAME_ATOM_TYPE_SUMMARY_COLUMNS,
            ),
            request=request,
            frame_indices=np.asarray(output_frames, dtype=int),
            iterations=np.asarray(output_iterations, dtype=int),
            time_values=(np.asarray(output_times, dtype=float) if have_all_times else None),
            detail_csv_path=str(detail_path) if detail_path is not None else None,
            charge_matrix_path=retained_matrix_path,
            charge_matrix_shape=(len(output_frames), max_atoms),
            atom_numbers=np.asarray(reported_atoms, dtype=int),
            atom_types=[atom_type_by_number.get(number, "") for number in reported_atoms],
            matrix_columns=np.asarray(
                [matrix_column_by_atom[number] for number in reported_atoms],
                dtype=int,
            ),
            baseline_charges=baseline_array,
            detail_row_count=detail_row_count,
        )


__all__ = [
    "DETAIL_COLUMNS",
    "FRAME_ATOM_TYPE_SUMMARY_COLUMNS",
    "FRAME_SUMMARY_COLUMNS",
    "SUMMARY_COLUMNS",
    "DynamicChargeChangeRequest",
    "DynamicChargeChangeResult",
    "DynamicChargeChangeTask",
    "calculate_dynamic_charge_changes",
]
