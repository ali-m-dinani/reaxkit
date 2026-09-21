"""Extended XYZ export for three-folded wurtzite polarity results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis.electrostatics.electrostatics import _field_component_series
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.neighbors import (
    _charges_for_request,
    _formal_charge_map,
    _frame_cell,
    _frame_iteration,
    _frame_labels,
    _select_frames,
    _source_frame,
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
from reaxkit.domain.data_models import ElectrostaticsData, TrajectoryData
from reaxkit.engine.common.generators.extended_xyz_generator import ExtendedXYZFrame, ExtendedXYZWriter
from reaxkit.presentation.specs import PresentationSpec


@dataclass
class PolarityExtendedXYZRequest(WurtzitePolarityRequest):
    precision: int = 8
    include_electric_field: bool = False
    field_direction: str = "z"
    _output_path: Optional[str] = None


@dataclass
class PolarityExtendedXYZResult(BaseResult):
    table: pd.DataFrame
    request: PolarityExtendedXYZRequest
    output_path: str
    frame_indices: np.ndarray
    iterations: np.ndarray
    polarity_result: WurtzitePolarityResult


def _validate_export_request(request: PolarityExtendedXYZRequest) -> None:
    if int(request.precision) < 1:
        raise ValueError("precision must be at least 1.")
    if request.field_direction not in {"x", "y", "z"}:
        raise ValueError("field_direction must be one of: x, y, z.")
    if not request._output_path:
        raise ValueError("An Extended XYZ output path is required.")


def _electric_field_metadata(data, request: PolarityExtendedXYZRequest, iteration: int):
    if not request.include_electric_field:
        return {}
    if not isinstance(data, ElectrostaticsData) or data.electric_field is None:
        raise ValueError(
            "include_electric_field requires electric-field data (for ReaxFF, provide fort.78)."
        )
    values = _field_component_series(
        data.electric_field,
        component=f"field_{request.field_direction}",
        target_iters=np.asarray([iteration], dtype=int),
    )
    return {
        "electric_field": float(np.asarray(values)[0]) * float(const("electric_field_VA_to_MVcm")),
        "electric_field_direction": request.field_direction,
        "electric_field_units": "MV/cm",
    }


def _all_atom_charges(data, request: PolarityExtendedXYZRequest, frame: int) -> np.ndarray:
    trajectory, charges = _charges_for_request(data, request)
    if charges is not None:
        return np.asarray(charges[frame], dtype=float)
    labels = _frame_labels(trajectory, frame)
    formal = _formal_charge_map(request)
    missing = sorted({str(label) for label in labels if str(label).casefold() not in formal})
    if missing:
        raise ValueError(
            "The Extended XYZ charge column requires an explicit formal charge for every "
            f"species present; missing: {', '.join(missing)}."
        )
    return np.asarray([formal[str(label).casefold()] for label in labels], dtype=float)


def polarity_extended_xyz_frame(
    data,
    request: PolarityExtendedXYZRequest,
    polarity_table: pd.DataFrame,
    *,
    data_frame_index: int = 0,
    output_frame_index: int | None = None,
) -> ExtendedXYZFrame:
    """Build a species-preserving frame with three-folded polarity properties."""

    trajectory, _ = _trajectory_and_charges(data)
    positions = np.asarray(trajectory.positions[data_frame_index], dtype=float)
    labels = _frame_labels(trajectory, data_frame_index).astype(str)
    atom_ids = np.asarray(trajectory.atom_ids, dtype=int)
    valid = np.isfinite(positions).all(axis=1)
    polarity = np.zeros(len(atom_ids), dtype=int)
    eta = np.full(len(atom_ids), np.nan)
    delta = np.full(len(atom_ids), np.nan)
    is_site = np.zeros(len(atom_ids), dtype=int)
    has_three = np.zeros(len(atom_ids), dtype=int)
    has_apical = np.zeros(len(atom_ids), dtype=int)
    proton = np.zeros(len(atom_ids), dtype=int)
    id_to_index = {int(atom_id): index for index, atom_id in enumerate(atom_ids)}
    for _, source in polarity_table.iterrows():
        index = id_to_index.get(int(source["site_atom_id"]))
        if index is None:
            continue
        polarity[index] = int(source["polarity"])
        eta[index] = float(source["eta_c (e*angstrom)"])
        delta[index] = float(source["delta_eff (angstrom)"])
        is_site[index] = 1
        has_three[index] = int(bool(source["has_three_basal_neighbors"]))
        has_apical[index] = int(bool(source["has_apical_neighbor"]))
        proton[index] = int(bool(source["has_proton_within_cutoff"]))

    lattice = _frame_cell(trajectory, request, data_frame_index)
    iteration = _frame_iteration(trajectory, data_frame_index)
    frame_number = (
        _source_frame(trajectory, data_frame_index)
        if output_frame_index is None else int(output_frame_index)
    )
    return ExtendedXYZFrame(
        species=labels[valid].tolist(), positions=positions[valid],
        properties={
            "atom_number": atom_ids[valid],
            "charge": _all_atom_charges(data, request, data_frame_index)[valid],
            "polarity": polarity[valid], "eta_c": eta[valid], "delta_eff": delta[valid],
            "is_polarity_site": is_site[valid],
            "has_three_basal_neighbors": has_three[valid],
            "has_apical_neighbor": has_apical[valid],
            "has_proton_within_cutoff": proton[valid],
        },
        frame=frame_number, iteration=iteration, lattice=lattice,
        pbc=tuple(bool(value) for value in request.periodic) if lattice is not None else None,
        metadata=_electric_field_metadata(data, request, iteration),
    )


def _result(
    request: PolarityExtendedXYZRequest,
    frames: Sequence[int],
    iterations: Sequence[int],
    atom_rows: Sequence[int],
    polarity_result: WurtzitePolarityResult,
) -> PolarityExtendedXYZResult:
    path = str(Path(request._output_path).resolve())
    return PolarityExtendedXYZResult(
        table=pd.DataFrame([{
            "output_path": path, "frames_written": len(frames),
            "atom_rows_written": int(sum(atom_rows)),
        }]),
        request=request, output_path=path,
        frame_indices=np.asarray(frames, dtype=int),
        iterations=np.asarray(iterations, dtype=int), polarity_result=polarity_result,
    )


@register_task(
    "write-three-folded-trajectory-with-polarity",
    label="Write Three-folded Trajectory with Polarity",
)
class PolarityExtendedXYZTask(AnalysisTask):
    """Write an OVITO-compatible trajectory with basal-only site polarity."""

    required_data = TrajectoryData
    supports_selective_streaming = False
    VERSION = "1"

    def required_data_for(self, request: PolarityExtendedXYZRequest, args: dict | None = None):
        if request.include_electric_field:
            return ElectrostaticsData
        return required_wurtzite_data_type(request, args)

    @staticmethod
    def required_data_fields_for(request: PolarityExtendedXYZRequest, _args: dict) -> tuple[str, ...]:
        fields = ["trajectory"]
        if request.charge_source != "formal" or request.include_electric_field:
            fields.append("charges")
        if request.include_electric_field:
            fields.append("electric_field")
        return tuple(fields)

    @staticmethod
    def recommended_presentations(
        _result: PolarityExtendedXYZResult, _payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        return [PresentationSpec(renderer="table", label="Three-folded polarity trajectory", view_type="table")]

    def run(self, data, request: PolarityExtendedXYZRequest, reporter=None):
        _validate_export_request(request)
        trajectory, _ = _trajectory_and_charges(data)
        selected = _select_frames(np.asarray(trajectory.positions).shape[0], request)
        polarity_result = calculate_polarity_from_trajectory(data, request)
        output_frames: list[int] = []
        iterations: list[int] = []
        atom_rows: list[int] = []
        with ExtendedXYZWriter(request._output_path, precision=request.precision) as writer:
            for progress, frame in enumerate(selected, start=1):
                site_table = polarity_result.table[
                    polarity_result.table["frame_index"].astype(int).eq(frame)
                ]
                extended = polarity_extended_xyz_frame(
                    data, request, site_table, data_frame_index=frame,
                    output_frame_index=_source_frame(trajectory, frame),
                )
                writer.write_frame(extended)
                output_frames.append(int(extended.frame))
                iterations.append(int(extended.iteration))
                atom_rows.append(len(extended.species))
                if callable(reporter):
                    reporter("write", progress, len(selected), "Writing three-folded polarity Extended XYZ")
        return _result(request, output_frames, iterations, atom_rows, polarity_result)


__all__ = [
    "PolarityExtendedXYZRequest", "PolarityExtendedXYZResult",
    "PolarityExtendedXYZTask", "polarity_extended_xyz_frame",
]
