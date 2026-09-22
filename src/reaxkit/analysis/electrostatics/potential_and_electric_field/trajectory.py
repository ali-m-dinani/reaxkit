"""Extended XYZ export containing calculated local potential and electric field."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ElectrostaticsData
from reaxkit.engine.common.generators.extended_xyz_generator import ExtendedXYZFrame, ExtendedXYZWriter
from reaxkit.presentation.specs import PresentationSpec

from .analysis import (
    PotentialElectricFieldRequest,
    PotentialElectricFieldResult,
    _cell,
    _labels,
    _source_frame,
    calculate_potential_and_field,
    combine_results,
    material_midpoint,
)


@dataclass
class PotentialElectricFieldTrajectoryRequest(PotentialElectricFieldRequest):
    precision: int = 8
    _output_path: Optional[str] = None


@dataclass
class PotentialElectricFieldTrajectoryResult(BaseResult):
    table: pd.DataFrame
    request: PotentialElectricFieldTrajectoryRequest
    output_path: str
    frame_indices: np.ndarray
    iterations: np.ndarray
    electrostatics_result: PotentialElectricFieldResult


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_") or "probe"


def _extended_frame(data: ElectrostaticsData, request, result: PotentialElectricFieldResult) -> ExtendedXYZFrame:
    trajectory = data.trajectory
    positions = np.asarray(trajectory.positions[0], dtype=float)
    labels = _labels(trajectory, 0).astype(str)
    charges = np.asarray(data.charges.charges, dtype=float)[0]
    valid = np.isfinite(positions).all(axis=1) & np.isfinite(charges) & np.asarray([bool(v.strip()) for v in labels])
    properties: dict[str, object] = {"atom_number": np.asarray(trajectory.atom_ids)[valid], "charge": charges[valid]}
    average = result.table.sort_values("atom_index")
    for label in ("internal", "external", "total_local"):
        properties[f"{label}_potential_V"] = average[f"{label}_potential (V)"].to_numpy()
        properties[f"{label}_electric_field_V_per_A"] = average[
            [f"{label}_electric_field_{axis} (V/angstrom)" for axis in "xyz"]
        ].to_numpy()
        properties[f"{label}_electric_field_MV_per_cm"] = average[
            [f"{label}_electric_field_{axis} (MV/cm)" for axis in "xyz"]
        ].to_numpy()
    for probe, table in result.probe_tables.items():
        name = _safe_name(probe); table = table.sort_values("atom_index")
        for label in ("internal", "total_local"):
            properties[f"{label}_potential_{name}_V"] = table[f"{label}_potential (V)"].to_numpy()
            properties[f"{label}_electric_field_{name}_V_per_A"] = table[
                [f"{label}_electric_field_{axis} (V/angstrom)" for axis in "xyz"]
            ].to_numpy()
    source = int(result.frame_indices[0]); iteration = int(result.iterations[0])
    cell = _cell(trajectory, request, 0)
    reference = average[[f"potential_reference_{axis} (angstrom)" for axis in "xyz"]].iloc[0].to_numpy(float)
    configured_mode = str(request.potential_reference_mode).strip().lower()
    reference_mode = ("fixed-midpoint" if configured_mode == "fixed-midpoint" else
                      "fixed-position" if request.potential_reference_position is not None else
                      "frame-midpoint")
    return ExtendedXYZFrame(labels[valid].tolist(), positions[valid], properties, frame=source,
                            iteration=iteration, lattice=cell,
                            pbc=tuple(bool(v) for v in request.periodic) if cell is not None else None,
                            metadata={"local_potential_units": "V", "local_field_units": "V/angstrom",
                                      "local_probe": "equal_species_average",
                                      "potential_reference_mode": reference_mode,
                                      "potential_reference_x": float(reference[0]),
                                      "potential_reference_y": float(reference[1]),
                                      "potential_reference_z": float(reference[2])})


def _summary(request, output_path: str, results: Sequence[PotentialElectricFieldResult], rows: int):
    combined = combine_results(results, request)
    return PotentialElectricFieldTrajectoryResult(
        pd.DataFrame([{"output_path": str(Path(output_path).resolve()), "frames_written": len(results),
                       "atom_rows_written": rows}]), request, str(Path(output_path).resolve()),
        combined.frame_indices, combined.iterations, combined)


def _slice_frame_dataclass(value, frame: int, n_frames: int):
    if value is None:
        return None
    updates = {}
    for item in fields(value):
        raw = getattr(value, item.name)
        if isinstance(raw, np.ndarray) and raw.ndim >= 1 and raw.shape[0] == n_frames:
            updates[item.name] = raw[frame:frame + 1]
    return replace(value, **updates)


@register_task("write-trajectory-with-potential-and-electric-field", label="Write Local Electrostatics Trajectory")
class PotentialElectricFieldTrajectoryTask(AnalysisTask):
    required_data = ElectrostaticsData
    supports_selective_streaming = True
    VERSION = "1"

    @staticmethod
    def required_data_fields_for(_request, _args):
        return ("trajectory", "charges", "electric_field")

    @staticmethod
    def recommended_presentations(_result, _payload: dict[str, Any]):
        return [PresentationSpec(renderer="table", label="Local electrostatics trajectory", view_type="table")]

    def run(self, data, request, reporter=None):
        if not request._output_path or int(request.precision) < 1:
            raise ValueError("A trajectory output path and precision >= 1 are required.")
        trajectory = data.trajectory
        selected = list(range(len(trajectory.positions))) if request.frames is None else list(request.frames)
        selected = selected[::int(request.every)]
        fixed_reference = (None if request.potential_reference_position is None
                           else tuple(request.potential_reference_position))
        if (selected and str(request.potential_reference_mode).strip().lower() == "fixed-midpoint"
                and fixed_reference is None):
            first_positions = np.asarray(trajectory.positions[selected[0]], dtype=float)
            fixed_reference = tuple(material_midpoint(first_positions[np.isfinite(first_positions).all(axis=1)]))
        results = []
        n_frames = len(trajectory.positions)
        with ExtendedXYZWriter(request._output_path, precision=request.precision) as writer:
            for count, frame in enumerate(selected, 1):
                one = ElectrostaticsData(
                    trajectory=replace(trajectory, positions=np.asarray(trajectory.positions)[frame:frame+1],
                        simulation=_slice_frame_dataclass(trajectory.simulation, frame, n_frames),
                        iterations=None if trajectory.iterations is None else np.asarray(trajectory.iterations)[frame:frame+1],
                        atom_labels=None if trajectory.atom_labels is None else np.asarray(trajectory.atom_labels)[frame:frame+1],
                        source_frame_indices=np.asarray([_source_frame(trajectory, frame)])),
                    charges=replace(data.charges, charges=np.asarray(data.charges.charges)[frame:frame+1],
                                    simulation=_slice_frame_dataclass(data.charges.simulation, frame, n_frames),
                                    total_charge=None if data.charges.total_charge is None else np.asarray(data.charges.total_charge)[frame:frame+1],
                                    iterations=None if data.charges.iterations is None else np.asarray(data.charges.iterations)[frame:frame+1]),
                    electric_field=data.electric_field)
                request_values = {**vars(request), "frames": [0], "every": 1}
                if fixed_reference is not None:
                    request_values["potential_reference_position"] = fixed_reference
                local_request = PotentialElectricFieldTrajectoryRequest(**request_values)
                result = calculate_potential_and_field(one, local_request, preserve_source_indices=True)
                writer.write_frame(_extended_frame(one, local_request, result)); results.append(result)
                if callable(reporter): reporter("write", count, len(selected), "Writing local electrostatics trajectory")
        return _summary(request, request._output_path, results, sum(len(v.table) for v in results))

    def run_stream(self, frames, request, reporter=None):
        if not request._output_path or int(request.precision) < 1:
            raise ValueError("A trajectory output path and precision >= 1 are required.")
        requested = None if request.frames is None else set(int(v) for v in request.frames[::int(request.every)])
        progress_total = len(requested) if requested is not None else 0
        results, rows, seen = [], 0, set()
        fixed_reference = (None if request.potential_reference_position is None
                           else tuple(request.potential_reference_position))
        output = Path(request._output_path)
        try:
            with ExtendedXYZWriter(output, precision=request.precision) as writer:
                for count, data in enumerate(frames, 1):
                    source = _source_frame(data.trajectory, 0); seen.add(source)
                    if source in requested if requested is not None else source % int(request.every) == 0:
                        request_values = {**vars(request), "frames": [0], "every": 1}
                        if str(request.potential_reference_mode).strip().lower() == "fixed-midpoint":
                            if fixed_reference is None:
                                positions = np.asarray(data.trajectory.positions[0], dtype=float)
                                fixed_reference = tuple(material_midpoint(positions[np.isfinite(positions).all(axis=1)]))
                            request_values["potential_reference_position"] = fixed_reference
                        local_request = PotentialElectricFieldTrajectoryRequest(**request_values)
                        result = calculate_potential_and_field(data, local_request, preserve_source_indices=True)
                        extended = _extended_frame(data, local_request, result)
                        writer.write_frame(extended); results.append(result); rows += len(extended.species)
                    if callable(reporter):
                        reporter(
                            "stream",
                            count,
                            progress_total,
                            "Calculating and writing local electrostatics",
                        )
        except Exception:
            output.unlink(missing_ok=True); raise
        if requested is not None and requested - seen:
            output.unlink(missing_ok=True); raise ValueError(f"Requested frame(s) not found: {sorted(requested - seen)}.")
        return _summary(request, str(output), results, rows)


__all__ = ["PotentialElectricFieldTrajectoryRequest", "PotentialElectricFieldTrajectoryResult",
           "PotentialElectricFieldTrajectoryTask"]
