"""Relate selected-atom dynamic charges to an applied electric field."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis.electrostatics.electrostatics import _field_component_series
from reaxkit.analysis.ferroelectrics.dynamic_charge import _atom_identity
from reaxkit.core.platform.constants import const
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ChargeData, ElectricFieldData
from reaxkit.presentation.specs import PresentationSpec

FieldDirection = Literal["x", "y", "z"]
TABLE_COLUMNS = [
    "frame",
    "atom_number",
    "atom_type",
    "charge",
    "delta_charge",
    "electric_field",
]


@dataclass
class ChargeFieldRequest(BaseRequest):
    """Select atoms, frames, and the applied-field component to correlate."""

    atom_numbers: Sequence[int] = dc_field(
        default_factory=tuple,
        metadata={"label": "Atom numbers", "help": "One or more atom numbers to analyze."},
    )
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={"label": "Frames", "help": "Optional source-frame indices."},
    )
    every: int = dc_field(
        default=1,
        metadata={"label": "Every", "help": "Keep every Nth frame.", "min": 1},
    )
    field_direction: FieldDirection = dc_field(
        default="z",
        metadata={"label": "Field direction", "choices": ["x", "y", "z"]},
    )
    reference_frame: int = dc_field(default=0, repr=False)
    _expected_frames: Optional[int] = None


@dataclass
class ChargeFieldResult(BaseResult):
    """Selected-atom charge rows aligned with applied electric-field samples."""

    table: pd.DataFrame
    request: ChargeFieldRequest
    frame_indices: np.ndarray
    iterations: np.ndarray
    time_values: Optional[np.ndarray] = None
    baseline_charges: dict[int, float] = dc_field(default_factory=dict)

    @property
    def csv_tables(self) -> dict[str, pd.DataFrame]:
        return {"charge_vs_electric_field": self.table}


def _validate_request(request: ChargeFieldRequest) -> list[int]:
    atoms = list(dict.fromkeys(int(value) for value in request.atom_numbers))
    if not atoms:
        raise ValueError("At least one atom number must be selected.")
    if int(request.every) < 1:
        raise ValueError("every must be at least 1.")
    if request.field_direction not in {"x", "y", "z"}:
        raise ValueError("field_direction must be one of: x, y, z.")
    if int(request.reference_frame) != 0:
        raise ValueError("Charge changes currently use frame 0 as their reference.")
    return atoms


def _align_field(
        field_data: ElectricFieldData,
        request: ChargeFieldRequest,
        iterations: np.ndarray,
) -> np.ndarray:
    values = _field_component_series(
        field_data,
        component=f"field_{request.field_direction}",
        target_iters=np.asarray(iterations, dtype=int),
    )
    return np.asarray(values, dtype=float) * float(const("electric_field_VA_to_MVcm"))


def _result_from_rows(
        rows: list[dict[str, object]],
        frame_indices: list[int],
        iterations: list[int],
        time_values: list[float],
        *,
        have_all_times: bool,
        field_data: ElectricFieldData,
        request: ChargeFieldRequest,
        baseline_charges: dict[int, float],
) -> ChargeFieldResult:
    field = _align_field(field_data, request, np.asarray(iterations, dtype=int))
    field_by_frame = dict(zip(frame_indices, field.tolist()))
    for row in rows:
        atom_number = int(row["atom_number"])
        baseline = baseline_charges.get(atom_number, float("nan"))
        row["delta_charge"] = float(row["charge"]) - baseline
        row["electric_field"] = field_by_frame[int(row["frame"])]
    return ChargeFieldResult(
        table=pd.DataFrame(rows, columns=TABLE_COLUMNS),
        request=request,
        frame_indices=np.asarray(frame_indices, dtype=int),
        iterations=np.asarray(iterations, dtype=int),
        time_values=np.asarray(time_values, dtype=float) if have_all_times else None,
        baseline_charges=dict(baseline_charges),
    )


def calculate_charge_field_response(
        charges_data: ChargeData,
        field_data: ElectricFieldData,
        request: ChargeFieldRequest,
) -> ChargeFieldResult:
    """Build selected-atom charge rows using iteration-matched field samples."""

    selected_atoms = _validate_request(request)
    charges = np.asarray(charges_data.charges, dtype=float)
    if charges.ndim != 2:
        raise ValueError("ChargeData.charges must have shape (n_frames, n_atoms).")
    n_frames, n_atoms = charges.shape
    atom_numbers, atom_types = _atom_identity(charges_data, n_atoms)
    atom_to_index = {number: index for index, number in enumerate(atom_numbers)}
    missing = [number for number in selected_atoms if number not in atom_to_index]
    if missing:
        raise ValueError(f"Atom number(s) not found in ChargeData: {missing}.")

    requested = list(range(n_frames)) if request.frames is None else [int(v) for v in request.frames]
    invalid = [frame for frame in requested if frame < 0 or frame >= n_frames]
    if invalid:
        raise ValueError(f"Frame(s) not found in ChargeData: {invalid}.")
    frame_indices = requested[:: int(request.every)]
    all_iterations = (
        np.asarray(charges_data.iterations, dtype=int)
        if charges_data.iterations is not None
        else np.arange(n_frames, dtype=int)
    )
    if all_iterations.shape != (n_frames,):
        raise ValueError("ChargeData.iterations length must match its frame count.")

    rows: list[dict[str, object]] = []
    baseline_charges = {
        atom_number: float(charges[0, atom_to_index[atom_number]])
        for atom_number in selected_atoms
        if np.isfinite(charges[0, atom_to_index[atom_number]])
    }
    for frame in frame_indices:
        for atom_number in selected_atoms:
            atom_index = atom_to_index[atom_number]
            charge = float(charges[frame, atom_index])
            if np.isfinite(charge):
                rows.append(
                    {
                        "frame": frame,
                        "atom_number": atom_number,
                        "atom_type": str(atom_types[atom_index]),
                        "charge": charge,
                    }
                )

    times: list[float] = []
    have_all_times = False
    if charges_data.simulation is not None and charges_data.simulation.time is not None:
        all_times = np.asarray(charges_data.simulation.time, dtype=float)
        if all_times.shape != (n_frames,):
            raise ValueError("ChargeData.simulation.time length must match its frame count.")
        times = all_times[frame_indices].tolist()
        have_all_times = True
    return _result_from_rows(
        rows,
        frame_indices,
        all_iterations[frame_indices].tolist(),
        times,
        have_all_times=have_all_times,
        field_data=field_data,
        request=request,
        baseline_charges=baseline_charges,
    )


@register_task("get_charge_vs_electric_field", label="Charge vs Electric Field")
class ChargeFieldTask(AnalysisTask):
    """Stream charges for selected atoms and align them with ``fort.78`` by iteration."""

    required_data = ChargeData
    VERSION = "4"
    from reaxkit.core.runtime.execution_contracts import TaskCapabilities, ExecutionShape
    execution_capabilities = TaskCapabilities(
        shape=ExecutionShape.REFERENCE_FRAME_MAP, reference_frames=(0,), needs_reference=True,
        supports_selective_frames=True, estimated_frame_bytes=1024 * 1024,
    )

    def __init__(self, electric_field: ElectricFieldData | None = None):
        self.electric_field = electric_field

    @staticmethod
    def recommended_presentations(
            _result: ChargeFieldResult,
            _payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        return [PresentationSpec(renderer="table", label="Charge vs field", view_type="table")]

    def _field_data(self) -> ElectricFieldData:
        if self.electric_field is None:
            raise ValueError("Charge-field analysis requires ElectricFieldData.")
        return self.electric_field

    def run(self, data: ChargeData, request: ChargeFieldRequest, reporter=None) -> ChargeFieldResult:
        _ = reporter
        return calculate_charge_field_response(data, self._field_data(), request)

    def run_stream(self, frames, request: ChargeFieldRequest, reporter=None) -> ChargeFieldResult:
        selected_atoms = _validate_request(request)
        requested = None if request.frames is None else [int(v) for v in request.frames][:: int(request.every)]
        requested_set = set(requested or ())
        seen_frames: set[int] = set()
        seen_atoms: set[int] = set()
        rows: list[dict[str, object]] = []
        frame_indices: list[int] = []
        iterations: list[int] = []
        times: list[float] = []
        have_all_times = True
        processed = 0
        baseline_charges: dict[int, float] | None = None

        for stream_index, data in enumerate(frames):
            processed += 1
            source_values = (data.metadata or {}).get("source_frame_indices")
            source_frame = (
                int(np.asarray(source_values).reshape(-1)[0])
                if source_values is not None
                else stream_index
            )
            seen_frames.add(source_frame)
            if source_frame == 0:
                baseline_values = np.asarray(data.charges, dtype=float)
                if baseline_values.shape[0] != 1:
                    raise ValueError("Streamed ChargeData must contain exactly one frame.")
                baseline_atoms, _ = _atom_identity(data, baseline_values.shape[1])
                baseline_index = {
                    number: index for index, number in enumerate(baseline_atoms)
                }
                baseline_charges = {
                    atom_number: float(baseline_values[0, baseline_index[atom_number]])
                    for atom_number in selected_atoms
                    if atom_number in baseline_index
                       and np.isfinite(baseline_values[0, baseline_index[atom_number]])
                }
            keep = (
                source_frame in requested_set
                if requested is not None
                else source_frame % int(request.every) == 0
            )
            if keep:
                charges = np.asarray(data.charges, dtype=float)
                if charges.shape[0] != 1:
                    raise ValueError("Streamed ChargeData must contain exactly one frame.")
                atom_numbers, atom_types = _atom_identity(data, charges.shape[1])
                atom_to_index = {number: index for index, number in enumerate(atom_numbers)}
                iteration = (
                    int(np.asarray(data.iterations).reshape(-1)[0])
                    if data.iterations is not None
                    else source_frame
                )
                frame_indices.append(source_frame)
                iterations.append(iteration)
                frame_times = data.simulation.time if data.simulation is not None else None
                if frame_times is None:
                    have_all_times = False
                else:
                    times.append(float(np.asarray(frame_times).reshape(-1)[0]))
                for atom_number in selected_atoms:
                    atom_index = atom_to_index.get(atom_number)
                    if atom_index is None:
                        continue
                    charge = float(charges[0, atom_index])
                    if not np.isfinite(charge):
                        continue
                    seen_atoms.add(atom_number)
                    rows.append(
                        {
                            "frame": source_frame,
                            "atom_number": atom_number,
                            "atom_type": str(atom_types[atom_index]),
                            "charge": charge,
                        }
                    )
            if callable(reporter):
                reporter(
                    "stream",
                    processed,
                    int(request._expected_frames or 0),
                    "Reading selected-atom charges",
                )

        if requested is not None:
            missing_frames = [frame for frame in requested if frame not in seen_frames]
            if missing_frames:
                raise ValueError(f"Requested frame(s) not found in ChargeData: {missing_frames}.")
        missing_atoms = [number for number in selected_atoms if number not in seen_atoms]
        if missing_atoms:
            raise ValueError(f"Atom number(s) not found in ChargeData: {missing_atoms}.")
        if baseline_charges is None:
            raise ValueError("Frame 0 is required to calculate delta_charge.")
        if callable(reporter):
            reporter("stream", processed, processed, "Finished reading selected-atom charges")
        return _result_from_rows(
            rows,
            frame_indices,
            iterations,
            times,
            have_all_times=have_all_times,
            field_data=self._field_data(),
            request=request,
            baseline_charges=baseline_charges,
        )


__all__ = [
    "TABLE_COLUMNS",
    "ChargeFieldRequest",
    "ChargeFieldResult",
    "ChargeFieldTask",
    "calculate_charge_field_response",
]
