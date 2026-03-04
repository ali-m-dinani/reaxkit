"""Generic time-series analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.alias import normalize_choice, resolve_alias_from_columns
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import (
    ChargeData,
    ElectricFieldData,
    EregimeData,
    MolecularAnalysisData,
    PartialEnergyData,
    RestraintData,
    SimulationData,
    TrajectoryData,
)
from reaxkit.presentation.convert import convert_xaxis


def _frame_indices(n_frames: int, frames: Optional[Sequence[int]], every: int) -> list[int]:
    idx = list(range(n_frames)) if frames is None else [int(i) for i in frames]
    return [i for i in idx if 0 <= i < n_frames][:: max(1, int(every))]


@dataclass
class Series:
    x: np.ndarray
    y: np.ndarray
    label: str


@dataclass
class TimeSeriesResult(BaseResult):
    series: list[Series]
    x_label: str
    y_label: str
    table: Optional[pd.DataFrame] = None
    metadata: dict | None = None


@dataclass
class SimulationScalarSeriesRequest(BaseRequest):
    field: str
    frames: Optional[Sequence[int]] = None
    every: int = 1


@dataclass
class TrajectoryCoordinateSeriesRequest(BaseRequest):
    atom_ids: Optional[Sequence[int]] = None
    atom_types: Optional[Sequence[str]] = None
    dims: Sequence[str] = ("x",)
    frames: Optional[Sequence[int]] = None
    every: int = 1


@dataclass
class CellDimensionsRequest(BaseRequest):
    fields: Sequence[str] = ("a", "b", "c")
    frames: Optional[Sequence[int]] = None
    every: int = 1


@dataclass
class ChargeSeriesRequest(BaseRequest):
    atom_ids: Sequence[int]
    frames: Optional[Sequence[int]] = None
    every: int = 1


@dataclass
class ElectricFieldSeriesRequest(BaseRequest):
    components: Sequence[str]
    field_kind: Literal["applied", "energy", "auto"] = "auto"
    frames: Optional[Sequence[int]] = None
    every: int = 1


@dataclass
class EregimeSeriesRequest(BaseRequest):
    field: str
    frames: Optional[Sequence[int]] = None
    every: int = 1


@dataclass
class PartialEnergySeriesRequest(BaseRequest):
    components: Optional[Sequence[str]] = None
    frames: Optional[Sequence[int]] = None
    every: int = 1


@dataclass
class RestraintSeriesRequest(BaseRequest):
    fields: Optional[Sequence[str]] = None
    restraint_index: Optional[int] = None
    dropna_rows: bool = False
    frames: Optional[Sequence[int]] = None
    every: int = 1


@dataclass
class MolecularFrequencySeriesRequest(BaseRequest):
    molecules: Sequence[str]
    frames: Optional[Sequence[int]] = None
    every: int = 1


@dataclass
class MolecularTotalsSeriesRequest(BaseRequest):
    quantities: Sequence[str] = ("total_molecules", "total_atoms", "total_molecular_mass")
    xaxis: Literal["iter", "frame", "time"] = "iter"
    control_file: str = "control"
    frames: Optional[Sequence[int]] = None
    every: int = 1


def _simulation_field_array(data: SimulationData, field: str) -> tuple[np.ndarray, str]:
    key = str(field).strip()
    if key == "potential_energy":
        arr = data.potential_energy
        label = "potential_energy"
    elif key in {"volume", "V"}:
        arr = data.volume
        label = "volume"
    elif key in {"temperature", "T"}:
        arr = data.temperature
        label = "temperature"
    elif key in {"pressure", "P"}:
        arr = data.pressure
        label = "pressure"
    elif key in {"density", "D"}:
        arr = data.density
        label = "density"
    elif key in {"elapsed_time", "elap_time"}:
        arr = data.elapsed_time
        label = "elapsed_time"
    elif key == "num_of_atoms":
        arr = data.num_of_atoms
        label = "num_of_atoms"
    elif key in {"a", "b", "c"}:
        if data.cell_lengths is None:
            arr = None
        else:
            j = {"a": 0, "b": 1, "c": 2}[key]
            arr = np.asarray(data.cell_lengths, dtype=float)[:, j]
        label = key
    elif key in {"alpha", "beta", "gamma"}:
        if data.cell_angles is None:
            arr = None
        else:
            j = {"alpha": 0, "beta": 1, "gamma": 2}[key]
            arr = np.asarray(data.cell_angles, dtype=float)[:, j]
        label = key
    else:
        raise KeyError(
            f"Unsupported simulation field {field!r}. "
            "Choose from: potential_energy, volume, temperature, pressure, density, elapsed_time, num_of_atoms, a, b, c, alpha, beta, gamma."
        )

    if arr is None:
        raise ValueError(f"Simulation field {field!r} is not available in loaded data.")
    return np.asarray(arr), label


@register_task("simulation_series")
class SimulationScalarSeriesTask(AnalysisTask):
    """Build a scalar time series from ``SimulationData``."""

    required_data = SimulationData

    def run(self, data: SimulationData, request: SimulationScalarSeriesRequest, reporter=None) -> TimeSeriesResult:
        values, label = _simulation_field_array(data, request.field)
        n_frames = int(values.shape[0])
        frame_idx = _frame_indices(n_frames, request.frames, request.every)
        iterations = (
            np.asarray(data.iterations, dtype=int)
            if data.iterations is not None
            else np.arange(n_frames, dtype=int)
        )
        if iterations.shape[0] != n_frames:
            raise ValueError("SimulationData.iterations length must match simulation scalar length.")

        series = [
            Series(
                x=iterations[frame_idx],
                y=np.asarray(values[frame_idx], dtype=float),
                label=label,
            )
        ]
        return TimeSeriesResult(
            series=series,
            x_label="iter",
            y_label=label,
            metadata={"frame_index": np.asarray(frame_idx, dtype=int), "iterations": iterations[frame_idx]},
        )


@register_task("trajectory_coordinate_series")
class TrajectoryCoordinateSeriesTask(AnalysisTask):
    """Build coordinate time series for one or more atoms."""

    required_data = TrajectoryData

    def run(self, data: TrajectoryData, request: TrajectoryCoordinateSeriesRequest, reporter=None) -> TimeSeriesResult:
        dims = tuple(str(d).lower() for d in request.dims if str(d).lower() in {"x", "y", "z"})
        if not dims:
            raise ValueError("TrajectoryCoordinateSeriesRequest.dims must include at least one of: x, y, z.")
        positions = np.asarray(data.positions, dtype=float)
        n_frames, n_atoms = positions.shape[:2]
        frame_idx = _frame_indices(n_frames, request.frames, request.every)
        iterations = (
            np.asarray(data.iterations, dtype=int)
            if data.iterations is not None
            else np.arange(n_frames, dtype=int)
        )
        if iterations.shape[0] != n_frames:
            raise ValueError("TrajectoryData.iterations length must match number of frames.")

        if request.atom_ids is not None:
            atom_ids = [int(a) for a in request.atom_ids]
            atom_indices = []
            for atom_id in atom_ids:
                atom_idx = int(atom_id) - 1
                if not (0 <= atom_idx < n_atoms):
                    raise ValueError(f"atom_id {atom_id} out of range 1..{n_atoms}.")
                atom_indices.append(atom_idx)
        elif request.atom_types:
            chosen = {str(t) for t in request.atom_types}
            atom_indices = [i for i, elem in enumerate(data.elements) if str(elem) in chosen]
            atom_ids = [int(data.atom_ids[i]) for i in atom_indices]
        else:
            atom_indices = list(range(n_atoms))
            atom_ids = [int(a) for a in data.atom_ids]

        rows = []
        out: list[Series] = []
        dim_to_col = {"x": 0, "y": 1, "z": 2}
        for atom_id, atom_idx in zip(atom_ids, atom_indices):
            atom_type = str(data.elements[atom_idx]) if atom_idx < len(data.elements) else ""
            for dim in dims:
                col = dim_to_col[dim]
                values = positions[frame_idx, atom_idx, col]
                out.append(
                    Series(
                        x=iterations[frame_idx],
                        y=np.asarray(values, dtype=float),
                        label=f"atom[{atom_id}].{dim}",
                    )
                )
            for rel_i, fi in enumerate(frame_idx):
                rec = {
                    "frame_index": int(fi),
                    "iter": int(iterations[fi]),
                    "atom_id": int(atom_id),
                    "atom_type": atom_type,
                }
                for dim in dims:
                    rec[dim] = float(positions[fi, atom_idx, dim_to_col[dim]])
                rows.append(rec)

        table = pd.DataFrame(rows).sort_values(["frame_index", "atom_id"]).reset_index(drop=True)

        return TimeSeriesResult(
            series=out,
            x_label="iter",
            y_label="coordinate",
            table=table,
            metadata={"frame_index": np.asarray(frame_idx, dtype=int), "iterations": iterations[frame_idx]},
        )


@register_task("cell_dimensions")
class CellDimensionsTask(AnalysisTask):
    """Build cell-dimension time series from ``SimulationData``."""

    required_data = SimulationData

    def run(self, data: SimulationData, request: CellDimensionsRequest, reporter=None) -> TimeSeriesResult:
        if data.cell_lengths is not None:
            n_frames = len(data.cell_lengths)
        elif data.cell_angles is not None:
            n_frames = len(data.cell_angles)
        elif data.iterations is not None:
            n_frames = len(data.iterations)
        else:
            n_frames = 0
        frame_idx = _frame_indices(
            n_frames,
            request.frames,
            request.every,
        )
        iterations = (
            np.asarray(data.iterations, dtype=int)
            if data.iterations is not None
            else np.arange(n_frames, dtype=int)
        )
        tables: list[Series] = []
        for field in request.fields:
            values, label = _simulation_field_array(data, str(field))
            tables.append(
                Series(
                    x=iterations[frame_idx],
                    y=np.asarray(values[frame_idx], dtype=float),
                    label=label,
                )
            )
        return TimeSeriesResult(
            series=tables,
            x_label="iter",
            y_label="cell_dimension",
            metadata={"frame_index": np.asarray(frame_idx, dtype=int), "iterations": iterations[frame_idx]},
        )


@register_task("charge_series")
class ChargeSeriesTask(AnalysisTask):
    """Build charge time series for one or more atoms."""

    required_data = ChargeData

    def run(self, data: ChargeData, request: ChargeSeriesRequest, reporter=None) -> TimeSeriesResult:
        charges = np.asarray(data.charges, dtype=float)
        if charges.ndim != 2:
            raise ValueError("ChargeData.charges must have shape (n_frames, n_atoms).")

        n_frames, n_atoms = charges.shape
        frame_idx = _frame_indices(n_frames, request.frames, request.every)
        iterations = (
            np.asarray(data.iterations, dtype=int).reshape(-1)
            if data.iterations is not None
            else np.arange(n_frames, dtype=int)
        )
        if iterations.shape[0] != n_frames:
            raise ValueError("ChargeData.iterations length must match number of frames.")

        if data.simulation is not None and data.simulation.atom_ids is not None:
            available_atom_ids = [int(a) for a in data.simulation.atom_ids]
        else:
            available_atom_ids = list(range(1, n_atoms + 1))
        if len(available_atom_ids) != n_atoms:
            raise ValueError("ChargeData atom_ids length must match number of atoms.")

        if data.simulation is not None and data.simulation.elements is not None:
            elements = [str(e) for e in data.simulation.elements]
        else:
            elements = [""] * n_atoms

        atom_id_to_idx = {atom_id: i for i, atom_id in enumerate(available_atom_ids)}
        rows: list[dict[str, object]] = []
        out: list[Series] = []
        for atom_id in request.atom_ids:
            atom_id_int = int(atom_id)
            if atom_id_int not in atom_id_to_idx:
                raise ValueError(f"atom_id {atom_id_int} not found in ChargeData.")
            atom_idx = atom_id_to_idx[atom_id_int]
            y = np.asarray(charges[frame_idx, atom_idx], dtype=float)
            out.append(Series(x=iterations[frame_idx], y=y, label=f"charge[{atom_id_int}]"))
            atom_type = elements[atom_idx] if atom_idx < len(elements) else ""
            for rel_i, fi in enumerate(frame_idx):
                rows.append(
                    {
                        "frame_index": int(fi),
                        "iter": int(iterations[fi]),
                        "atom_id": atom_id_int,
                        "atom_type": atom_type,
                        "charge": float(y[rel_i]),
                    }
                )

        table = pd.DataFrame(rows)
        if not table.empty:
            table = table.sort_values(["frame_index", "atom_id"], kind="stable").reset_index(drop=True)

        return TimeSeriesResult(
            series=out,
            x_label="iter",
            y_label="charge",
            table=table,
            metadata={"frame_index": np.asarray(frame_idx, dtype=int), "iterations": iterations[frame_idx]},
        )


def _electric_field_group(
    data: ElectricFieldData,
    field_kind: str,
) -> tuple[np.ndarray, list[str], str]:
    kind = str(field_kind).lower()
    if kind == "applied":
        return (
            np.asarray(data.applied_field_values, dtype=float),
            [str(c) for c in data.applied_field_components],
            "applied_field",
        )
    if kind == "energy":
        return (
            np.asarray(data.field_energy_values, dtype=float),
            [str(c) for c in data.field_energy_components],
            "field_energy",
        )
    return np.empty((0, 0), dtype=float), [], ""


def _resolve_eregime_field(df: pd.DataFrame, name: str) -> str:
    if not name:
        raise ValueError("Column name is empty.")

    canonical = normalize_choice(name)
    hit = resolve_alias_from_columns(df.columns, canonical)
    if hit:
        return hit

    if canonical in {"field", "field_dir"}:
        for i in range(1, 16):
            cand = f"{canonical}{i}" if canonical != "field_dir" else f"field_dir{i}"
            hit = resolve_alias_from_columns(df.columns, cand)
            if hit:
                return hit

    raise ValueError(
        f"Column '{name}' not found (after alias resolution). "
        f"Available columns: {list(df.columns)}"
    )


def _partial_energy_frame(data: PartialEnergyData) -> pd.DataFrame:
    df = pd.DataFrame({"iter": pd.Series(data.iterations, dtype=int)})
    if data.components:
        df = pd.concat([df, pd.DataFrame(data.values, columns=list(data.components))], axis=1)
    return df


def _resolve_partial_energy_components(
    df: pd.DataFrame,
    components: Optional[Sequence[str]],
) -> list[str]:
    available = [str(c) for c in df.columns if str(c) != "iter"]
    if components is None or len(components) == 0:
        return available

    resolved: list[str] = []
    for component in components:
        canonical = normalize_choice(str(component))
        actual = resolve_alias_from_columns(available, canonical)
        if actual is None:
            raise KeyError(
                f"Requested component '{component}' not found in partial-energy data. "
                f"Available components: {available}"
            )
        if actual not in resolved:
            resolved.append(actual)
    return resolved


def _restraint_frame(data: RestraintData) -> pd.DataFrame:
    df = pd.DataFrame({"iter": pd.Series(data.iterations, dtype=int)})
    n_restraints = int(data.metadata.get("n_restraints", 0)) if data.metadata else 0
    if data.restraint_energy is not None:
        df["E_res"] = pd.Series(data.restraint_energy, dtype=float)
    else:
        df["E_res"] = pd.Series([pd.NA] * len(df))
    if data.potential_energy is not None:
        df["E_pot"] = pd.Series(data.potential_energy, dtype=float)
    else:
        df["E_pot"] = pd.Series([pd.NA] * len(df))
    for i in range(n_restraints):
        target_col = f"r{i + 1}_target"
        actual_col = f"r{i + 1}_actual"
        if i < data.target_values.shape[1]:
            df[target_col] = pd.Series(data.target_values[:, i], dtype=float)
        else:
            df[target_col] = pd.Series([pd.NA] * len(df))
        if i < data.actual_values.shape[1]:
            df[actual_col] = pd.Series(data.actual_values[:, i], dtype=float)
        else:
            df[actual_col] = pd.Series([pd.NA] * len(df))
    return df


def _resolve_restraint_field(df: pd.DataFrame, requested: str) -> str:
    cols = [str(c) for c in df.columns]
    key = str(requested).strip().lower()
    if not key:
        raise KeyError("Requested restraint column is empty.")

    if "restraint" in key or key.startswith("r"):
        parts = key.replace("_", " ").split()
        idx = None
        for part in parts:
            if part.isdigit():
                idx = int(part)
                break
        if idx is not None:
            if "target" in parts:
                col = f"r{idx}_target"
            elif "actual" in parts:
                col = f"r{idx}_actual"
            else:
                raise KeyError(f"Restraint column must specify target or actual: '{requested}'")
            if col in cols:
                return col
            raise KeyError(
                f"Resolved '{requested}' to '{col}', but that column is not available. "
                f"Available columns: {cols}"
            )

    canonical = normalize_choice(str(requested))
    hit = resolve_alias_from_columns(cols, canonical)
    if hit is not None:
        return hit
    if str(requested) in cols:
        return str(requested)
    raise KeyError(
        f"Could not resolve requested restraint column '{requested}'. "
        f"Available columns: {cols}"
    )


def _resolve_restraint_fields(df: pd.DataFrame, request: RestraintSeriesRequest) -> list[str]:
    if request.restraint_index is not None:
        idx = int(request.restraint_index)
        if idx < 1:
            raise ValueError("restraint_index must be >= 1.")
        return [f"r{idx}_target", f"r{idx}_actual"]
    if request.fields is None or len(request.fields) == 0:
        raise ValueError("RestraintSeriesRequest requires either fields or restraint_index.")
    resolved: list[str] = []
    for field in request.fields:
        actual = _resolve_restraint_field(df, str(field))
        if actual not in resolved:
            resolved.append(actual)
    return resolved


@register_task("electric_field_series")
class ElectricFieldSeriesTask(AnalysisTask):
    """Build time series for applied-field or field-energy components."""

    required_data = ElectricFieldData

    def run(self, data: ElectricFieldData, request: ElectricFieldSeriesRequest, reporter=None) -> TimeSeriesResult:
        frame_values: np.ndarray
        component_names: list[str]
        y_label: str

        if str(request.field_kind).lower() == "auto":
            requested = [str(c) for c in request.components]
            applied_values, applied_names, _ = _electric_field_group(data, "applied")
            energy_values, energy_names, _ = _electric_field_group(data, "energy")
            if all(comp in applied_names for comp in requested):
                frame_values, component_names, y_label = applied_values, applied_names, "applied_field"
            elif all(comp in energy_names for comp in requested):
                frame_values, component_names, y_label = energy_values, energy_names, "field_energy"
            else:
                available = sorted(set(applied_names + energy_names))
                raise KeyError(
                    f"Requested components {requested} are not available in a single field group. "
                    f"Available components: {available}"
                )
        else:
            frame_values, component_names, y_label = _electric_field_group(data, request.field_kind)

        if frame_values.ndim == 1:
            frame_values = frame_values.reshape(-1, 1)
        if frame_values.ndim != 2:
            raise ValueError("ElectricFieldData values must be 1D or 2D.")

        n_frames = frame_values.shape[0]
        iterations = (
            np.asarray(data.sampled_field_iterations, dtype=int).reshape(-1)
            if data.sampled_field_iterations is not None
            else np.arange(n_frames, dtype=int)
        )
        if iterations.shape[0] != n_frames:
            raise ValueError("ElectricFieldData.sampled_field_iterations length must match number of samples.")
        frame_idx = _frame_indices(n_frames, request.frames, request.every)

        rows: list[dict[str, object]] = []
        out: list[Series] = []
        for comp in request.components:
            comp_name = str(comp)
            if comp_name not in component_names:
                raise KeyError(f"Electric field component '{comp_name}' not found in {component_names}.")
            j = component_names.index(comp_name)
            y = np.asarray(frame_values[frame_idx, j], dtype=float)
            out.append(Series(x=iterations[frame_idx], y=y, label=comp_name))
            for rel_i, fi in enumerate(frame_idx):
                rows.append(
                    {
                        "frame_index": int(fi),
                        "iter": int(iterations[fi]),
                        "component": comp_name,
                        "value": float(y[rel_i]),
                    }
                )

        table = pd.DataFrame(rows)
        if not table.empty:
            table = table.sort_values(["frame_index", "component"], kind="stable").reset_index(drop=True)

        return TimeSeriesResult(
            series=out,
            x_label="iter",
            y_label=y_label,
            table=table,
            metadata={"frame_index": np.asarray(frame_idx, dtype=int), "iterations": iterations[frame_idx]},
        )


@register_task("eregime_series")
class EregimeSeriesTask(AnalysisTask):
    """Build iteration-based series for one eregime field column."""

    required_data = EregimeData

    def run(self, data: EregimeData, request: EregimeSeriesRequest, reporter=None) -> TimeSeriesResult:
        iterations = np.asarray(data.iterations, dtype=int).reshape(-1)
        field_zones = np.asarray(data.field_zones, dtype=int).reshape(-1)
        field_dir = np.asarray(data.field_dir, dtype=object).reshape(-1)
        field = np.asarray(data.field, dtype=float).reshape(-1)
        df = pd.DataFrame(
            {
                "iter": iterations,
                "field_zones": field_zones,
                "field_dir": field_dir,
                "field": field,
            }
        )
        field_col = _resolve_eregime_field(df, request.field)
        n_frames = iterations.shape[0]
        frame_idx = _frame_indices(n_frames, request.frames, request.every)

        series = [
            Series(
                x=iterations[frame_idx],
                y=df.iloc[frame_idx][field_col].to_numpy(),
                label=str(field_col),
            )
        ]
        table = pd.DataFrame(
            {
                "frame_index": np.asarray(frame_idx, dtype=int),
                "iter": iterations[frame_idx],
                "field": str(field_col),
                "value": df.iloc[frame_idx][field_col].to_numpy(),
            }
        )
        return TimeSeriesResult(
            series=series,
            x_label="iter",
            y_label=str(field_col),
            table=table,
            metadata={"frame_index": np.asarray(frame_idx, dtype=int), "iterations": iterations[frame_idx]},
        )


@register_task("partial_energy_series")
class PartialEnergySeriesTask(AnalysisTask):
    """Build iteration-based series for one or more fort.73 energy components."""

    required_data = PartialEnergyData

    def run(self, data: PartialEnergyData, request: PartialEnergySeriesRequest, reporter=None) -> TimeSeriesResult:
        df = _partial_energy_frame(data)
        iterations = np.asarray(data.iterations, dtype=int).reshape(-1)
        n_frames = iterations.shape[0]
        if len(df) != n_frames:
            raise ValueError("PartialEnergyData.values length must match PartialEnergyData.iterations length.")

        components = _resolve_partial_energy_components(df, request.components)
        frame_idx = _frame_indices(n_frames, request.frames, request.every)

        rows: list[dict[str, object]] = []
        series: list[Series] = []
        for component in components:
            y = pd.to_numeric(df.iloc[frame_idx][component], errors="coerce").to_numpy(dtype=float)
            series.append(Series(x=iterations[frame_idx], y=y, label=str(component)))
            for rel_i, fi in enumerate(frame_idx):
                rows.append(
                    {
                        "frame_index": int(fi),
                        "iter": int(iterations[fi]),
                        "component": str(component),
                        "value": float(y[rel_i]),
                    }
                )

        table = pd.DataFrame(rows)
        if not table.empty:
            table = table.sort_values(["frame_index", "component"], kind="stable").reset_index(drop=True)

        return TimeSeriesResult(
            series=series,
            x_label="iter",
            y_label="partial_energy",
            table=table,
            metadata={"frame_index": np.asarray(frame_idx, dtype=int), "iterations": iterations[frame_idx]},
        )


@register_task("restraint_series")
class RestraintSeriesTask(AnalysisTask):
    """Build iteration-based series for fort.76 restraint data."""

    required_data = RestraintData

    def run(self, data: RestraintData, request: RestraintSeriesRequest, reporter=None) -> TimeSeriesResult:
        df = _restraint_frame(data)
        iterations = np.asarray(data.iterations, dtype=int).reshape(-1)
        n_frames = iterations.shape[0]
        if len(df) != n_frames:
            raise ValueError("RestraintData arrays must all align with RestraintData.iterations.")

        fields = _resolve_restraint_fields(df, request)
        frame_idx = _frame_indices(n_frames, request.frames, request.every)
        selected = df.iloc[frame_idx].copy()
        selected["iter"] = iterations[frame_idx]
        selected = selected.loc[:, ["iter", *[c for c in selected.columns if c != "iter"]]]
        if request.dropna_rows:
            non_iter = [c for c in fields if c != "iter"]
            if non_iter:
                selected = selected.dropna(axis=0, how="all", subset=non_iter).reset_index(drop=True)

        selected_iters = pd.to_numeric(selected["iter"], errors="coerce").to_numpy(dtype=int)
        series: list[Series] = []
        for field in fields:
            y = pd.to_numeric(selected[field], errors="coerce").to_numpy(dtype=float)
            series.append(Series(x=selected_iters, y=y, label=str(field)))

        return TimeSeriesResult(
            series=series,
            x_label="iter",
            y_label="restraint",
            table=selected.loc[:, ["iter", *fields]].copy(),
            metadata={"frame_index": np.asarray(frame_idx, dtype=int), "iterations": selected_iters},
        )


@register_task("molecular_frequency_series")
class MolecularFrequencySeriesTask(AnalysisTask):
    """Build molecular-frequency time series for one or more molecular formulas."""

    required_data = MolecularAnalysisData

    def run(
        self,
        data: MolecularAnalysisData,
        request: MolecularFrequencySeriesRequest,
        reporter=None,
    ) -> TimeSeriesResult:
        df = data.molecular_species.copy()
        iterations = np.asarray(data.iterations, dtype=int).reshape(-1)
        n_frames = iterations.shape[0]
        frame_idx = _frame_indices(n_frames, request.frames, request.every)
        selected_iters = iterations[frame_idx]

        rows: list[dict[str, object]] = []
        out: list[Series] = []
        for molecule in request.molecules:
            sub = df[df["molecular_formula"] == str(molecule)][["iter", "freq"]].copy()
            sub["iter"] = pd.to_numeric(sub["iter"], errors="coerce").astype(int)
            sub["freq"] = pd.to_numeric(sub["freq"], errors="coerce").astype(float)
            freq_by_iter = pd.Series(sub["freq"].to_numpy(dtype=float), index=sub["iter"].to_numpy(dtype=int))
            y = freq_by_iter.reindex(selected_iters, fill_value=0.0).to_numpy(dtype=float)
            out.append(Series(x=selected_iters, y=y, label=str(molecule)))
            for rel_i, fi in enumerate(frame_idx):
                rows.append(
                    {
                        "frame_index": int(fi),
                        "iter": int(selected_iters[rel_i]),
                        "molecular_formula": str(molecule),
                        "freq": float(y[rel_i]),
                    }
                )

        table = pd.DataFrame(rows)
        if not table.empty:
            table = table.sort_values(["frame_index", "molecular_formula"], kind="stable").reset_index(drop=True)

        return TimeSeriesResult(
            series=out,
            x_label="iter",
            y_label="molecular_frequency",
            table=table,
            metadata={"frame_index": np.asarray(frame_idx, dtype=int), "iterations": selected_iters},
        )


@register_task("molecular_totals_series")
class MolecularTotalsSeriesTask(AnalysisTask):
    """Build total molecule/atom/mass time series from MolecularAnalysisData.totals."""

    required_data = MolecularAnalysisData

    def run(
        self,
        data: MolecularAnalysisData,
        request: MolecularTotalsSeriesRequest,
        reporter=None,
    ) -> TimeSeriesResult:
        df = data.totals.copy()
        if df.empty:
            return TimeSeriesResult(
                series=[],
                x_label=str(request.xaxis),
                y_label="molecular_totals",
                table=pd.DataFrame(),
                metadata=None,
            )

        iterations = pd.to_numeric(df["iter"], errors="coerce").to_numpy(dtype=int)
        n_frames = len(iterations)
        frame_idx = _frame_indices(n_frames, request.frames, request.every)
        if not frame_idx:
            return TimeSeriesResult(
                series=[],
                x_label=str(request.xaxis),
                y_label="molecular_totals",
                table=pd.DataFrame(),
                metadata={"frame_index": np.asarray([], dtype=int), "iterations": np.asarray([], dtype=int)},
            )

        xaxis = str(request.xaxis).lower()
        if xaxis == "iter":
            x_vals = iterations[frame_idx]
            xlabel = "iter"
        else:
            converted, xlabel = convert_xaxis(iterations[frame_idx], xaxis, control_file=request.control_file)
            x_vals = np.asarray(converted)

        quantities = [q for q in request.quantities if q in df.columns]
        rows: list[dict[str, object]] = []
        out: list[Series] = []
        for quantity in quantities:
            y = pd.to_numeric(df.iloc[frame_idx][quantity], errors="coerce").to_numpy(dtype=float)
            out.append(Series(x=np.asarray(x_vals), y=y, label=str(quantity)))
            for rel_i, fi in enumerate(frame_idx):
                rows.append(
                    {
                        "frame_index": int(fi),
                        "iter": int(iterations[fi]),
                        "quantity": str(quantity),
                        "value": float(y[rel_i]),
                    }
                )

        table = pd.DataFrame(rows)
        if not table.empty:
            table.insert(2, xlabel, np.tile(np.asarray(x_vals), len(quantities)))
            table = table.sort_values(["frame_index", "quantity"], kind="stable").reset_index(drop=True)

        return TimeSeriesResult(
            series=out,
            x_label=xlabel,
            y_label="molecular_totals",
            table=table,
            metadata={"frame_index": np.asarray(frame_idx, dtype=int), "iterations": iterations[frame_idx]},
        )


__all__ = [
    "Series",
    "TimeSeriesResult",
    "SimulationScalarSeriesRequest",
    "SimulationScalarSeriesTask",
    "TrajectoryCoordinateSeriesRequest",
    "TrajectoryCoordinateSeriesTask",
    "CellDimensionsRequest",
    "CellDimensionsTask",
    "ChargeSeriesRequest",
    "ChargeSeriesTask",
    "ElectricFieldSeriesRequest",
    "ElectricFieldSeriesTask",
    "EregimeSeriesRequest",
    "EregimeSeriesTask",
    "PartialEnergySeriesRequest",
    "PartialEnergySeriesTask",
    "RestraintSeriesRequest",
    "RestraintSeriesTask",
    "MolecularFrequencySeriesRequest",
    "MolecularFrequencySeriesTask",
    "MolecularTotalsSeriesRequest",
    "MolecularTotalsSeriesTask",
]
