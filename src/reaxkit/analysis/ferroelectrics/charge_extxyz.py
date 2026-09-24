"""Export charge and frame-zero charge changes as an Extended XYZ trajectory."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis.electrostatics.electrostatics import _field_component_series
from reaxkit.core.platform.constants import const
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ElectrostaticsData
from reaxkit.engine.common.generators.extended_xyz_generator import (
    ExtendedXYZFrame,
    ExtendedXYZWriter,
    lattice_from_lengths_angles,
)
from reaxkit.presentation.specs import PresentationSpec


@dataclass
class ChargeExtendedXYZRequest(BaseRequest):
    """Configure charge-property Extended XYZ generation."""

    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={"label": "Frames", "help": "Optional source-frame indices."},
    )
    every: int = dc_field(
        default=1,
        metadata={"label": "Every", "help": "Keep every Nth frame.", "min": 1},
    )
    precision: int = dc_field(
        default=8,
        metadata={"label": "Precision", "help": "Significant digits in the output.", "min": 1},
    )
    include_electric_field: bool = dc_field(
        default=False,
        metadata={
            "label": "Include electric field",
            "help": "Write an iteration-matched electric field into each frame header.",
        },
    )
    field_direction: str = dc_field(
        default="z",
        metadata={"label": "Field direction", "choices": ["x", "y", "z"]},
    )
    reference_frame: int = dc_field(default=0, repr=False)
    _output_path: Optional[str] = None
    _expected_frames: Optional[int] = None


@dataclass
class ChargeExtendedXYZResult(BaseResult):
    """Metadata describing a generated Extended XYZ trajectory."""

    table: pd.DataFrame
    request: ChargeExtendedXYZRequest
    output_path: str
    frame_indices: np.ndarray
    iterations: np.ndarray


def _validate_request(request: ChargeExtendedXYZRequest) -> None:
    if int(request.every) < 1:
        raise ValueError("every must be at least 1.")
    if int(request.precision) < 1:
        raise ValueError("precision must be at least 1.")
    if int(request.reference_frame) != 0:
        raise ValueError("Charge changes currently use frame 0 as their reference.")
    if request.field_direction not in {"x", "y", "z"}:
        raise ValueError("field_direction must be one of: x, y, z.")
    if not request._output_path:
        raise ValueError("An Extended XYZ output path is required.")


def _source_frame(data: ElectrostaticsData, fallback: int) -> int:
    source = data.trajectory.source_frame_indices
    return int(np.asarray(source).reshape(-1)[0]) if source is not None else int(fallback)


def _frame_iteration(data: ElectrostaticsData, frame_index: int = 0) -> int:
    iterations = data.charges.iterations
    if iterations is None:
        iterations = data.trajectory.iterations
    return (
        int(np.asarray(iterations).reshape(-1)[frame_index])
        if iterations is not None
        else frame_index
    )


def _baseline_by_atom(data: ElectrostaticsData, frame_index: int = 0) -> dict[int, float]:
    atom_ids = [int(value) for value in data.charges.simulation.atom_ids]
    charges = np.asarray(data.charges.charges[frame_index], dtype=float)
    return {
        atom_number: float(charge)
        for atom_number, charge in zip(atom_ids, charges, strict=False)
        if np.isfinite(charge)
    }


def charge_extended_xyz_frame(
        data: ElectrostaticsData,
        baseline: dict[int, float],
        *,
        data_frame_index: int = 0,
        source_frame: int = 0,
        metadata: Optional[dict[str, object]] = None,
) -> ExtendedXYZFrame:
    """Create one OVITO-ready frame from aligned trajectory and charge data."""

    trajectory = data.trajectory
    charges_data = data.charges
    positions = np.asarray(trajectory.positions[data_frame_index], dtype=float)
    charges = np.asarray(charges_data.charges[data_frame_index], dtype=float)
    atom_ids = np.asarray(charges_data.simulation.atom_ids, dtype=int)
    if positions.shape != (charges.size, 3) or atom_ids.size != charges.size:
        raise ValueError("Trajectory positions, atom ids, and charges must share one atom dimension.")
    labels = (
        np.asarray(trajectory.atom_labels[data_frame_index], dtype=object)
        if trajectory.atom_labels is not None
        else np.asarray(trajectory.elements, dtype=object)
    )
    valid = np.isfinite(positions).all(axis=1) & np.isfinite(charges)
    selected_ids = atom_ids[valid]
    selected_charges = charges[valid]
    delta_charges = np.asarray(
        [
            charge - baseline.get(int(atom_number), float("nan"))
            for atom_number, charge in zip(selected_ids, selected_charges, strict=False)
        ],
        dtype=float,
    )

    lattice = None
    simulation = trajectory.simulation
    if (
            simulation is not None
            and simulation.cell_lengths is not None
            and simulation.cell_angles is not None
    ):
        lattice = lattice_from_lengths_angles(
            np.asarray(simulation.cell_lengths, dtype=float)[data_frame_index],
            np.asarray(simulation.cell_angles, dtype=float)[data_frame_index],
        )
    return ExtendedXYZFrame(
        species=labels[valid].astype(str).tolist(),
        positions=positions[valid],
        properties={
            "atom_number": selected_ids,
            "charge": selected_charges,
            "delta_charge": delta_charges,
        },
        frame=int(source_frame),
        iteration=_frame_iteration(data, data_frame_index),
        lattice=lattice,
        pbc=(True, True, True) if lattice is not None else None,
        metadata=metadata or {},
    )


def _electric_field_metadata(
        data: ElectrostaticsData,
        request: ChargeExtendedXYZRequest,
        iteration: int,
) -> dict[str, object]:
    """Return the requested field component matched exactly by iteration."""

    if not request.include_electric_field:
        return {}
    if data.electric_field is None:
        raise ValueError(
            "--include-electric-field requires electric-field data (for ReaxFF, provide fort.78)."
        )
    values = _field_component_series(
        data.electric_field,
        component=f"field_{request.field_direction}",
        target_iters=np.asarray([iteration], dtype=int),
    )
    value = float(np.asarray(values, dtype=float)[0])
    value *= float(const("electric_field_VA_to_MVcm"))
    return {
        "electric_field": value,
        "electric_field_direction": request.field_direction,
        "electric_field_units": "MV/cm",
    }


def _result(
        request: ChargeExtendedXYZRequest,
        frames: list[int],
        iterations: list[int],
        atom_rows: list[int],
) -> ChargeExtendedXYZResult:
    output_path = str(Path(request._output_path).resolve())
    return ChargeExtendedXYZResult(
        table=pd.DataFrame(
            [
                {
                    "output_path": output_path,
                    "frames_written": len(frames),
                    "atom_rows_written": int(sum(atom_rows)),
                }
            ]
        ),
        request=request,
        output_path=output_path,
        frame_indices=np.asarray(frames, dtype=int),
        iterations=np.asarray(iterations, dtype=int),
    )


@register_task("write_trajectory_with_charges", label="Write Trajectory with Charges")
class ChargeExtendedXYZTask(AnalysisTask):
    """Write an OVITO-compatible charge trajectory with bounded memory."""

    required_data = ElectrostaticsData
    VERSION = "2"

    @staticmethod
    def required_data_fields_for(
            request: ChargeExtendedXYZRequest,
            _args: dict,
    ) -> tuple[str, ...]:
        fields = ["trajectory", "charges"]
        if request.include_electric_field:
            fields.append("electric_field")
        return tuple(fields)

    @staticmethod
    def recommended_presentations(
            _result: ChargeExtendedXYZResult,
            _payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        return [PresentationSpec(renderer="table", label="Extended XYZ", view_type="table")]

    def run(
            self,
            data: ElectrostaticsData,
            request: ChargeExtendedXYZRequest,
            reporter=None,
    ) -> ChargeExtendedXYZResult:
        _validate_request(request)
        n_frames = np.asarray(data.trajectory.positions).shape[0]
        requested = (
            list(range(n_frames))
            if request.frames is None
            else [int(value) for value in request.frames]
        )
        invalid = [frame for frame in requested if frame < 0 or frame >= n_frames]
        if invalid:
            raise ValueError(f"Frame(s) not found in trajectory: {invalid}.")
        selected = requested[:: int(request.every)]
        baseline = _baseline_by_atom(data, 0)
        frame_indices: list[int] = []
        iterations: list[int] = []
        atom_rows: list[int] = []
        with ExtendedXYZWriter(request._output_path, precision=request.precision) as writer:
            for progress_index, frame in enumerate(selected, start=1):
                extended = charge_extended_xyz_frame(
                    data,
                    baseline,
                    data_frame_index=frame,
                    source_frame=frame,
                    metadata=_electric_field_metadata(
                        data,
                        request,
                        _frame_iteration(data, frame),
                    ),
                )
                writer.write_frame(extended)
                frame_indices.append(frame)
                iterations.append(int(extended.iteration))
                atom_rows.append(len(extended.species))
                if callable(reporter):
                    reporter("write", progress_index, len(selected), "Writing charge Extended XYZ")
        return _result(request, frame_indices, iterations, atom_rows)

    def run_stream(
            self,
            frames,
            request: ChargeExtendedXYZRequest,
            reporter=None,
    ) -> ChargeExtendedXYZResult:
        _validate_request(request)
        requested = (
            None
            if request.frames is None
            else [int(value) for value in request.frames][:: int(request.every)]
        )
        requested_set = set(requested or ())
        baseline: dict[int, float] | None = None
        seen_frames: set[int] = set()
        frame_indices: list[int] = []
        iterations: list[int] = []
        atom_rows: list[int] = []
        processed = 0
        output_path = Path(request._output_path)

        try:
            with ExtendedXYZWriter(output_path, precision=request.precision) as writer:
                for stream_index, data in enumerate(frames):
                    processed += 1
                    source_frame = _source_frame(data, stream_index)
                    seen_frames.add(source_frame)
                    if source_frame == 0:
                        baseline = _baseline_by_atom(data)
                    keep = (
                        source_frame in requested_set
                        if requested is not None
                        else source_frame % int(request.every) == 0
                    )
                    if keep:
                        if baseline is None:
                            raise ValueError("Frame 0 must be read before charge deltas are generated.")
                        extended = charge_extended_xyz_frame(
                            data,
                            baseline,
                            source_frame=source_frame,
                            metadata=_electric_field_metadata(
                                data,
                                request,
                                _frame_iteration(data),
                            ),
                        )
                        writer.write_frame(extended)
                        frame_indices.append(source_frame)
                        iterations.append(int(extended.iteration))
                        atom_rows.append(len(extended.species))
                    if callable(reporter):
                        reporter(
                            "stream",
                            processed,
                            int(request._expected_frames or 0),
                            "Reading coordinates and charges; writing Extended XYZ",
                        )
        except Exception:
            output_path.unlink(missing_ok=True)
            raise

        if baseline is None:
            output_path.unlink(missing_ok=True)
            raise ValueError("Frame 0 is required to calculate delta_charge.")
        if requested is not None:
            missing = [frame for frame in requested if frame not in seen_frames]
            if missing:
                output_path.unlink(missing_ok=True)
                raise ValueError(f"Requested frame(s) not found in trajectory: {missing}.")
        if callable(reporter):
            reporter("stream", processed, processed, "Finished charge Extended XYZ")
        return _result(request, frame_indices, iterations, atom_rows)


__all__ = [
    "ChargeExtendedXYZRequest",
    "ChargeExtendedXYZResult",
    "ChargeExtendedXYZTask",
    "charge_extended_xyz_frame",
]
