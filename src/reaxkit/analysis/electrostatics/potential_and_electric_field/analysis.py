"""Registered ReaxKit analysis for ReaxFF local electrostatic observables."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ElectricFieldData, ElectrostaticsData
from reaxkit.presentation.specs import PresentationSpec

from .calculation import calculate_frame, reaxff_cell_matrix
from .parameters import ReaxFFCoulombParameters
from .physics import V_PER_ANGSTROM_TO_MV_PER_CM


@dataclass
class PotentialElectricFieldRequest(BaseRequest):
    """Configure shielded Coulomb potential and electric-field evaluation."""

    lower_taper_radius: float = 0.0
    upper_taper_radius: float = 10.0
    gamma_by_symbol: Mapping[str, float] = field(default_factory=dict)
    parameter_source: str = ""
    probe_elements: Sequence[str] = ()
    periodic: Sequence[bool] = (True, True, True)
    cell_lengths: Optional[Sequence[float]] = None
    cell_angles: Sequence[float] = (90.0, 90.0, 90.0)
    frames: Optional[Sequence[int]] = None
    every: int = 1
    field_method: str = "analytic"
    field_step: float = 0.001
    disable_taper: bool = False
    potential_reference_mode: str = "frame-midpoint"
    potential_reference_position: Optional[Sequence[float]] = None


@dataclass
class PotentialElectricFieldResult(BaseResult):
    """Average-probe table plus species-probe and Coulomb tables."""

    table: pd.DataFrame
    request: PotentialElectricFieldRequest
    coulomb_table: pd.DataFrame
    totals: pd.DataFrame
    probe_tables: dict[str, pd.DataFrame]
    frame_indices: np.ndarray
    iterations: np.ndarray

    @property
    def csv_tables(self) -> dict[str, pd.DataFrame]:
        return {"probe_average": self.table, "coulomb_per_atom": self.coulomb_table,
                "coulomb_totals": self.totals, **{f"probe_{key}": value for key, value in self.probe_tables.items()}}


def _trajectory_parts(data: ElectrostaticsData):
    return data.trajectory, np.asarray(data.charges.charges, dtype=float)


def _labels(trajectory, frame: int) -> np.ndarray:
    return (np.asarray(trajectory.atom_labels[frame], dtype=object)
            if trajectory.atom_labels is not None else np.asarray(trajectory.elements, dtype=object))


def _source_frame(trajectory, frame: int) -> int:
    return (int(np.asarray(trajectory.source_frame_indices)[frame])
            if trajectory.source_frame_indices is not None else int(frame))


def _iteration(trajectory, frame: int) -> int:
    values = trajectory.iterations
    if values is None and trajectory.simulation is not None:
        values = trajectory.simulation.iterations
    return int(np.asarray(values)[frame]) if values is not None else _source_frame(trajectory, frame)


def _cell(trajectory, request: PotentialElectricFieldRequest, frame: int):
    if request.cell_lengths is not None:
        return reaxff_cell_matrix(request.cell_lengths, request.cell_angles)
    simulation = trajectory.simulation
    if simulation is None or simulation.cell_lengths is None:
        return None
    lengths = np.asarray(simulation.cell_lengths, dtype=float)[frame]
    angles = (np.asarray(simulation.cell_angles, dtype=float)[frame]
              if simulation.cell_angles is not None else np.full(3, 90.0))
    return reaxff_cell_matrix(lengths, angles)


def _parameters(request: PotentialElectricFieldRequest) -> ReaxFFCoulombParameters:
    return ReaxFFCoulombParameters(float(request.lower_taper_radius), float(request.upper_taper_radius),
                                   request.gamma_by_symbol, request.parameter_source)


def material_midpoint(positions: np.ndarray) -> np.ndarray:
    """Return the center of the occupied Cartesian bounding box."""
    xyz = np.asarray(positions, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("Finite atom positions with shape (atoms, 3) are required for the material midpoint.")
    return (np.min(xyz, axis=0) + np.max(xyz, axis=0)) / 2.0


def _external_field_at_iteration(field_data: ElectricFieldData | None, iteration: int,
                                 source_frame: int) -> np.ndarray:
    if field_data is None:
        raise ValueError("Local electrostatics requires an externally applied electric field, usually from fort.78.")
    values = np.asarray(field_data.applied_field_values, dtype=float)
    components = tuple(str(value) for value in field_data.applied_field_components)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or not len(values) or values.shape[1] != len(components):
        raise ValueError("Applied electric-field values and component labels are missing or inconsistent.")
    if not any(name in components for name in ("field_x", "field_y", "field_z")):
        raise ValueError("Applied electric-field data contains none of field_x, field_y, or field_z.")
    if field_data.sampled_field_iterations is None:
        row = int(source_frame)
        if row >= len(values):
            raise ValueError(f"No applied electric-field sample exists for frame {source_frame}.")
    else:
        sampled = np.asarray(field_data.sampled_field_iterations, dtype=int).reshape(-1)
        matches = np.flatnonzero(sampled == int(iteration))
        if not len(matches):
            raise ValueError(f"No applied electric-field sample exists for iteration {iteration}.")
        row = int(matches[-1])
    output = np.zeros(3, dtype=float)
    for axis, name in enumerate(("field_x", "field_y", "field_z")):
        if name in components:
            output[axis] = values[row, components.index(name)]
    if not np.isfinite(output).all():
        raise ValueError(f"Applied electric-field components are not finite at iteration {iteration}.")
    return output


def _reference_position(request: PotentialElectricFieldRequest, positions: np.ndarray,
                        fixed_midpoint: np.ndarray | None) -> np.ndarray:
    mode = str(request.potential_reference_mode).strip().lower()
    if mode not in {"frame-midpoint", "fixed-midpoint"}:
        raise ValueError("potential_reference_mode must be 'frame-midpoint' or 'fixed-midpoint'.")
    if request.potential_reference_position is not None:
        reference = np.asarray(request.potential_reference_position, dtype=float)
        if reference.shape != (3,) or not np.isfinite(reference).all():
            raise ValueError("potential_reference_position must contain three finite Cartesian coordinates.")
        return reference
    if mode == "fixed-midpoint" and fixed_midpoint is not None:
        return np.asarray(fixed_midpoint, dtype=float)
    return material_midpoint(positions)


def _probe_rows(base: dict[str, np.ndarray], probe: str, internal_potential: np.ndarray,
                internal_field: np.ndarray, external_field: np.ndarray,
                reference: np.ndarray) -> pd.DataFrame:
    external_field_rows = np.broadcast_to(np.asarray(external_field, dtype=float), internal_field.shape)
    external_potential = -(base["xyz"] - reference) @ external_field
    total_potential = internal_potential + external_potential
    total_field = internal_field + external_field_rows
    output: dict[str, object] = {
        "frame_index": base["frame_index"], "iter": base["iter"],
        "atom_index": base["atom_index"], "atom_id": base["atom_id"],
        "atom_element": base["atom_element"], "x (angstrom)": base["x"],
        "y (angstrom)": base["y"], "z (angstrom)": base["z"],
        "charge (e)": base["charge"], "probe_element": probe,
        "potential_reference_x (angstrom)": reference[0],
        "potential_reference_y (angstrom)": reference[1],
        "potential_reference_z (angstrom)": reference[2],
        "internal_potential (V)": internal_potential,
        "external_potential (V)": external_potential,
        "total_local_potential (V)": total_potential,
    }
    for label, values in (("internal", internal_field), ("external", external_field_rows),
                          ("total_local", total_field)):
        magnitude = np.linalg.norm(values, axis=1)
        for axis, index in zip("xyz", range(3)):
            output[f"{label}_electric_field_{axis} (V/angstrom)"] = values[:, index]
        output[f"{label}_electric_field_magnitude (V/angstrom)"] = magnitude
        for axis, index in zip("xyz", range(3)):
            output[f"{label}_electric_field_{axis} (MV/cm)"] = values[:, index] * V_PER_ANGSTROM_TO_MV_PER_CM
        output[f"{label}_electric_field_magnitude (MV/cm)"] = magnitude * V_PER_ANGSTROM_TO_MV_PER_CM
    return pd.DataFrame(output)


def calculate_potential_and_field(data: ElectrostaticsData, request: PotentialElectricFieldRequest,
                                  *, preserve_source_indices: bool = False) -> PotentialElectricFieldResult:
    if int(request.every) < 1:
        raise ValueError("every must be at least 1.")
    trajectory, charges = _trajectory_parts(data)
    positions = np.asarray(trajectory.positions, dtype=float)
    if charges.shape != positions.shape[:2]:
        raise ValueError("Charge and trajectory arrays must have matching frame and atom dimensions.")
    selected = list(range(len(positions))) if request.frames is None else [int(value) for value in request.frames]
    selected = selected[::int(request.every)]
    invalid = [value for value in selected if value < 0 or value >= len(positions)]
    if invalid:
        raise ValueError(f"Frame(s) not found in trajectory: {invalid}.")
    parameters = _parameters(request)
    probe_tables: dict[str, list[pd.DataFrame]] = {}
    coulomb_frames: list[pd.DataFrame] = []
    totals: list[dict[str, object]] = []
    output_frames: list[int] = []
    iterations: list[int] = []
    resolved_probes: tuple[str, ...] | None = None
    fixed_midpoint: np.ndarray | None = None
    for frame in selected:
        xyz, labels = positions[frame], _labels(trajectory, frame)
        valid = np.isfinite(xyz).all(axis=1) & np.isfinite(charges[frame]) & np.asarray([bool(str(v).strip()) for v in labels])
        indices = np.flatnonzero(valid)
        frame_labels = tuple(str(value) for value in labels[valid])
        probes = tuple(request.probe_elements) or tuple(dict.fromkeys(frame_labels))
        if resolved_probes is None:
            resolved_probes = probes
        elif tuple(value.casefold() for value in probes) != tuple(value.casefold() for value in resolved_probes):
            raise ValueError("Automatic probe species changed between frames; specify probe_elements explicitly.")
        result = calculate_frame(xyz[valid], charges[frame, valid], frame_labels, parameters,
                                 probe_elements=resolved_probes, cell=_cell(trajectory, request, frame),
                                 periodic=request.periodic, field_method=request.field_method,
                                 field_step=float(request.field_step), disable_taper=bool(request.disable_taper))
        source = _source_frame(trajectory, frame) if preserve_source_indices else frame
        iteration = _iteration(trajectory, frame)
        external_field = _external_field_at_iteration(data.electric_field, iteration, source)
        if (str(request.potential_reference_mode).strip().lower() == "fixed-midpoint"
                and request.potential_reference_position is None and fixed_midpoint is None):
            fixed_midpoint = material_midpoint(xyz[valid])
        reference = _reference_position(request, xyz[valid], fixed_midpoint)
        ids = np.asarray(trajectory.atom_ids, dtype=int)[valid]
        base = {"frame_index": np.full(len(indices), source), "iter": np.full(len(indices), iteration),
                "atom_index": indices, "atom_id": ids, "atom_element": np.asarray(frame_labels),
                "x": xyz[valid, 0], "y": xyz[valid, 1], "z": xyz[valid, 2], "xyz": xyz[valid],
                "charge": charges[frame, valid]}
        coulomb_frames.append(pd.DataFrame({
            "frame_index": base["frame_index"], "iter": base["iter"], "atom_index": indices,
            "atom_id": ids, "atom_element": frame_labels, "x (angstrom)": base["x"],
            "y (angstrom)": base["y"], "z (angstrom)": base["z"], "charge (e)": base["charge"],
            "coulomb (kcal/mol)": result.per_atom_coulomb_kcal_per_mol,
        }))
        for probe_index, probe in enumerate(resolved_probes):
            probe_tables.setdefault(probe, []).append(_probe_rows(base, probe,
                result.internal_probe_potential_v[probe_index],
                result.internal_probe_field_v_per_angstrom[probe_index],
                external_field, reference))
        totals.append({"frame_index": source, "iter": iteration,
                       "coulomb (kcal/mol)": result.total_coulomb_kcal_per_mol,
                       "distinct_interactions": result.distinct_interactions,
                       "self_image_interactions": result.self_image_interactions})
        output_frames.append(source); iterations.append(iteration)
    joined = {label: pd.concat(frames, ignore_index=True) for label, frames in probe_tables.items()}
    if joined:
        stacked = pd.concat(joined.values(), ignore_index=True)
        keys = ["frame_index", "iter", "atom_index", "atom_id", "atom_element"]
        numeric = [column for column in stacked.columns if column not in {*keys, "probe_element"}]
        average = stacked.groupby(keys, as_index=False, sort=False)[numeric].mean()
        average["probe_element"] = "average"
        ordered = list(next(iter(joined.values())).columns)
        average = average.loc[:, ordered]
    else:
        average = pd.DataFrame()
    return PotentialElectricFieldResult(average, request,
        pd.concat(coulomb_frames, ignore_index=True) if coulomb_frames else pd.DataFrame(),
        pd.DataFrame(totals), joined, np.asarray(output_frames, dtype=int), np.asarray(iterations, dtype=int))


def combine_results(results: Sequence[PotentialElectricFieldResult], request) -> PotentialElectricFieldResult:
    def concat(attribute):
        values = [getattr(result, attribute) for result in results]
        return pd.concat(values, ignore_index=True) if values else pd.DataFrame()
    labels = tuple(dict.fromkeys(label for result in results for label in result.probe_tables))
    probes = {label: pd.concat([result.probe_tables[label] for result in results if label in result.probe_tables], ignore_index=True)
              for label in labels}
    return PotentialElectricFieldResult(concat("table"), request, concat("coulomb_table"), concat("totals"), probes,
        np.concatenate([result.frame_indices for result in results]) if results else np.empty(0, dtype=int),
        np.concatenate([result.iterations for result in results]) if results else np.empty(0, dtype=int))


@register_task("get-potential-and-electric-field", label="ReaxFF Potential and Electric Field")
class PotentialElectricFieldTask(AnalysisTask):
    required_data = ElectrostaticsData
    supports_selective_streaming = True
    VERSION = "2"

    @staticmethod
    def required_data_fields_for(_request, _args) -> tuple[str, ...]:
        return ("trajectory", "charges", "electric_field")

    @staticmethod
    def recommended_presentations(_result, _payload: dict[str, Any]) -> list[PresentationSpec]:
        return [PresentationSpec(renderer="table", label="Local electrostatics", view_type="table")]

    def run(self, data, request, reporter=None):
        _ = reporter
        return calculate_potential_and_field(data, request)

    def run_stream(self, frames, request, reporter=None):
        requested = None if request.frames is None else set(int(value) for value in request.frames[::int(request.every)])
        progress_total = len(requested) if requested is not None else 0
        results, seen = [], set()
        fixed_reference = (None if request.potential_reference_position is None
                           else tuple(request.potential_reference_position))
        for count, data in enumerate(frames, start=1):
            trajectory = data.trajectory
            source = _source_frame(trajectory, 0)
            seen.add(source)
            keep = source in requested if requested is not None else source % int(request.every) == 0
            if keep:
                request_values = {**vars(request), "frames": [0], "every": 1}
                if str(request.potential_reference_mode).strip().lower() == "fixed-midpoint":
                    if fixed_reference is None:
                        valid = np.isfinite(trajectory.positions[0]).all(axis=1)
                        fixed_reference = tuple(material_midpoint(trajectory.positions[0][valid]))
                    request_values["potential_reference_position"] = fixed_reference
                    request_values["potential_reference_mode"] = "fixed-midpoint"
                local_request = PotentialElectricFieldRequest(**request_values)
                results.append(calculate_potential_and_field(data, local_request, preserve_source_indices=True))
            if callable(reporter):
                reporter(
                    "stream",
                    count,
                    progress_total,
                    "Calculating ReaxFF potential and electric field",
                )
        if requested is not None and requested - seen:
            raise ValueError(f"Requested frame(s) not found: {sorted(requested - seen)}.")
        return combine_results(results, request)


__all__ = ["PotentialElectricFieldRequest", "PotentialElectricFieldResult", "PotentialElectricFieldTask",
           "calculate_potential_and_field", "combine_results", "material_midpoint"]
