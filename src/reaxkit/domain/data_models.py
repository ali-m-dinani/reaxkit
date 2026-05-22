"""Engine-agnostic domain data models for analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd


def _as_1d(name: str, values: Any, *, dtype=None) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype) if dtype is not None else np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got shape={arr.shape}.")
    return arr


def _as_2d(name: str, values: Any, *, dtype=None) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype) if dtype is not None else np.asarray(values)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D; got shape={arr.shape}.")
    return arr


def _require_len(name: str, values: Any, expected: int) -> None:
    arr = np.asarray(values)
    if arr.shape[0] != expected:
        raise ValueError(f"{name} length ({arr.shape[0]}) must match expected ({expected}).")


def _ensure_int_list(name: str, values: Sequence[int]) -> list[int]:
    out = [int(v) for v in values]
    if len(set(out)) != len(out):
        raise ValueError(f"{name} contains duplicate values.")
    return out


@dataclass
class SimulationData:
    """Shared simulation-level identity data."""

    atom_ids: Sequence[int]
    iterations: Optional[np.ndarray] = None
    time: Optional[np.ndarray] = None
    elements: Optional[list[str]] = None
    num_of_atoms: Optional[np.ndarray] = None
    potential_energy: Optional[np.ndarray] = None
    volume: Optional[np.ndarray] = None
    temperature: Optional[np.ndarray] = None
    pressure: Optional[np.ndarray] = None
    density: Optional[np.ndarray] = None
    elapsed_time: Optional[np.ndarray] = None
    atom_type_nums: Optional[np.ndarray] = None   # (n_frames, n_atoms_max), padded with 0 when an atom is absent
    molecule_nums: Optional[np.ndarray] = None    # (n_frames, n_atoms_max), padded with 0 when an atom is absent
    cell_lengths: Optional[np.ndarray] = None  # (n_frames, 3): a, b, c
    cell_angles: Optional[np.ndarray] = None   # (n_frames, 3): alpha, beta, gamma

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        atom_ids = _ensure_int_list("SimulationData.atom_ids", self.atom_ids)
        n_atoms = len(atom_ids)
        if self.elements is not None and len(self.elements) != n_atoms:
            raise ValueError("SimulationData.elements length must match atom_ids length.")

        n_frames = None
        if self.iterations is not None:
            n_frames = len(_as_1d("SimulationData.iterations", self.iterations, dtype=int))

        for name in [
            "time",
            "num_of_atoms",
            "potential_energy",
            "volume",
            "temperature",
            "pressure",
            "density",
            "elapsed_time",
        ]:
            values = getattr(self, name)
            if values is None:
                continue
            _as_1d(f"SimulationData.{name}", values)
            if n_frames is None:
                n_frames = len(np.asarray(values))
            else:
                _require_len(f"SimulationData.{name}", values, n_frames)

        if self.cell_lengths is not None:
            arr = _as_2d("SimulationData.cell_lengths", self.cell_lengths, dtype=float)
            if arr.shape[1] != 3:
                raise ValueError("SimulationData.cell_lengths must have shape (n_frames, 3).")
            if n_frames is None:
                n_frames = arr.shape[0]
            elif arr.shape[0] != n_frames:
                raise ValueError("SimulationData.cell_lengths frame count must match iterations/series length.")

        if self.cell_angles is not None:
            arr = _as_2d("SimulationData.cell_angles", self.cell_angles, dtype=float)
            if arr.shape[1] != 3:
                raise ValueError("SimulationData.cell_angles must have shape (n_frames, 3).")
            if n_frames is None:
                n_frames = arr.shape[0]
            elif arr.shape[0] != n_frames:
                raise ValueError("SimulationData.cell_angles frame count must match iterations/series length.")

        if self.atom_type_nums is not None:
            arr = _as_2d("SimulationData.atom_type_nums", self.atom_type_nums)
            if arr.shape[1] != n_atoms:
                raise ValueError("SimulationData.atom_type_nums atom dimension must match atom_ids length.")
            if n_frames is not None and arr.shape[0] != n_frames:
                raise ValueError("SimulationData.atom_type_nums frame count must match iterations/series length.")

        if self.molecule_nums is not None:
            arr = _as_2d("SimulationData.molecule_nums", self.molecule_nums)
            if arr.shape[1] != n_atoms:
                raise ValueError("SimulationData.molecule_nums atom dimension must match atom_ids length.")
            if n_frames is not None and arr.shape[0] != n_frames:
                raise ValueError("SimulationData.molecule_nums frame count must match iterations/series length.")


@dataclass
class TrajectoryData:
    """Canonical trajectory model consumed by analysis tasks."""

    positions: np.ndarray  # (n_frames, n_atoms_max, 3), padded with NaN when an atom is absent
    elements: list[str]
    atom_ids: list[int]
    simulation: Optional[SimulationData] = None
    iterations: Optional[np.ndarray] = None
    atom_labels: Optional[np.ndarray] = None  # (n_frames, n_atoms_max) per-frame labels, padded with ""

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        pos = np.asarray(self.positions, dtype=float)
        if pos.ndim != 3 or pos.shape[2] != 3:
            raise ValueError("TrajectoryData.positions must have shape (n_frames, n_atoms, 3).")
        n_frames, n_atoms = pos.shape[:2]

        atom_ids = _ensure_int_list("TrajectoryData.atom_ids", self.atom_ids)
        if len(atom_ids) != n_atoms:
            raise ValueError("TrajectoryData.atom_ids length must match positions atom dimension.")
        if len(self.elements) != n_atoms:
            raise ValueError("TrajectoryData.elements length must match positions atom dimension.")

        if self.iterations is not None:
            arr = _as_1d("TrajectoryData.iterations", self.iterations, dtype=int)
            if arr.shape[0] != n_frames:
                raise ValueError("TrajectoryData.iterations length must match positions frame dimension.")

        if self.atom_labels is not None:
            labels = np.asarray(self.atom_labels, dtype=object)
            if labels.shape != (n_frames, n_atoms):
                raise ValueError("TrajectoryData.atom_labels must have shape (n_frames, n_atoms).")

        if self.simulation is not None:
            self.simulation.validate()
            if len(self.simulation.atom_ids) != n_atoms:
                raise ValueError("TrajectoryData.simulation.atom_ids length must match trajectory atom dimension.")


@dataclass
class GeometryData:
    """Canonical single-structure geometry model."""

    coordinates: pd.DataFrame = field(default_factory=pd.DataFrame)
    connectivity: pd.DataFrame = field(default_factory=pd.DataFrame)
    atom_ids: list[int] = field(default_factory=list)
    elements: list[str] = field(default_factory=list)
    descriptor: str = ""
    remark: str = ""
    lattice_parameters: Optional[dict[str, float | None]] = None
    simulation: Optional[SimulationData] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class ConnectivityData:
    """Canonical connectivity model.

    Notes
    -----
    ``connectivity`` stores neighbor links (graph edges) and may be a dense
    matrix or per-frame sparse list. Engine-agnostic connectivity analyses
    should prefer ``bond_orders``/``sum_bond_orders`` when bond strength
    information is required.
    """

    connectivity: Any = None
    bond_orders: Any = None
    sum_bond_orders: Optional[np.ndarray] = None  # (n_frames, n_atoms_max), padded with NaN when an atom is absent
    num_lone_pairs: Optional[np.ndarray] = None  # (n_frames, n_atoms_max), padded with NaN when an atom is absent
    num_of_bonds: Optional[np.ndarray] = None    # (n_frames,)
    total_bond_order: Optional[np.ndarray] = None
    total_lone_pairs: Optional[np.ndarray] = None
    total_bond_order_uncorrected: Optional[np.ndarray] = None
    atom_ids: Optional[Sequence[int]] = None
    elements: Optional[list[str]] = None
    simulation: Optional[SimulationData] = None
    iterations: Optional[np.ndarray] = None
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        n_frames = None
        n_atoms = None
        if self.sum_bond_orders is not None:
            arr = _as_2d("ConnectivityData.sum_bond_orders", self.sum_bond_orders, dtype=float)
            n_frames, n_atoms = arr.shape
        if self.num_lone_pairs is not None:
            arr = _as_2d("ConnectivityData.num_lone_pairs", self.num_lone_pairs, dtype=float)
            if n_frames is None:
                n_frames, n_atoms = arr.shape
            elif arr.shape != (n_frames, n_atoms):
                raise ValueError("ConnectivityData.num_lone_pairs shape must match sum_bond_orders shape.")

        if self.iterations is not None:
            it = _as_1d("ConnectivityData.iterations", self.iterations, dtype=int)
            if n_frames is None:
                n_frames = it.shape[0]
            elif it.shape[0] != n_frames:
                raise ValueError("ConnectivityData.iterations length must match frame count.")

        if self.atom_ids is not None:
            atom_ids = _ensure_int_list("ConnectivityData.atom_ids", self.atom_ids)
            if n_atoms is None:
                n_atoms = len(atom_ids)
            elif len(atom_ids) != n_atoms:
                raise ValueError("ConnectivityData.atom_ids length must match atom dimension.")

        if self.elements is not None and n_atoms is not None and len(self.elements) != n_atoms:
            raise ValueError("ConnectivityData.elements length must match atom dimension.")

        if self.simulation is not None:
            self.simulation.validate()


@dataclass
class ForceFieldParametersData:
    """Engine-agnostic force-field reference model."""

    general_parameters: pd.DataFrame
    atom_parameters: pd.DataFrame
    bond_parameters: pd.DataFrame
    off_diagonal_parameters: pd.DataFrame
    angle_parameters: pd.DataFrame
    torsion_parameters: pd.DataFrame
    hydrogen_bond_parameters: pd.DataFrame
    source: str = ""
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        for name in [
            "general_parameters",
            "atom_parameters",
            "bond_parameters",
            "off_diagonal_parameters",
            "angle_parameters",
            "torsion_parameters",
            "hydrogen_bond_parameters",
        ]:
            if not isinstance(getattr(self, name), pd.DataFrame):
                raise TypeError(f"ForceFieldParametersData.{name} must be a pandas DataFrame.")


@dataclass
class ForceFieldOptimizationProgressData:
    """Canonical force-field optimization error model parsed from ``fort.13``."""

    epochs: np.ndarray
    total_ff_error: np.ndarray
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        epochs = _as_1d("ForceFieldOptimizationProgressData.epochs", self.epochs, dtype=int)
        errs = _as_1d("ForceFieldOptimizationProgressData.total_ff_error", self.total_ff_error, dtype=float)
        if epochs.shape[0] != errs.shape[0]:
            raise ValueError("ForceFieldOptimizationProgressData arrays must have same length.")


@dataclass
class ForceFieldOptimizationDiagnosticData:
    """Canonical parameter-optimization diagnostic model parsed from ``fort.79``."""

    identifiers: np.ndarray
    value1: np.ndarray
    value2: np.ndarray
    value3: np.ndarray
    diff1: np.ndarray
    diff2: np.ndarray
    diff3: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    parabol_min: np.ndarray
    parabol_min_diff: np.ndarray
    value4: np.ndarray
    diff4: np.ndarray
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        fields = [
            "identifiers",
            "value1",
            "value2",
            "value3",
            "diff1",
            "diff2",
            "diff3",
            "a",
            "b",
            "c",
            "parabol_min",
            "parabol_min_diff",
            "value4",
            "diff4",
        ]
        lens = []
        for name in fields:
            arr = _as_1d(f"ForceFieldOptimizationDiagnosticData.{name}", getattr(self, name))
            lens.append(arr.shape[0])
        if len(set(lens)) > 1:
            raise ValueError("ForceFieldOptimizationDiagnosticData arrays must have identical length.")


@dataclass
class ForceFieldOptimizationReportData:
    """Canonical force-field optimization report model parsed from ``fort.99``."""

    linenos: np.ndarray
    sections: np.ndarray
    titles: np.ndarray
    ffield_values: np.ndarray
    qm_values: np.ndarray
    weights: np.ndarray
    errors: np.ndarray
    total_ff_error: np.ndarray
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        fields = ["linenos", "sections", "titles", "ffield_values", "qm_values", "weights", "errors", "total_ff_error"]
        lens = []
        for name in fields:
            arr = _as_1d(f"ForceFieldOptimizationReportData.{name}", getattr(self, name))
            lens.append(arr.shape[0])
        if len(set(lens)) > 1:
            raise ValueError("ForceFieldOptimizationReportData arrays must have identical length.")


@dataclass
class ForceFieldOptimizationTrainingSetData:
    """Canonical ReaxFF training-set model parsed from ``trainset`` files."""

    sections: tuple[str, ...] = ()
    charge: pd.DataFrame = field(default_factory=pd.DataFrame)
    heatfo: pd.DataFrame = field(default_factory=pd.DataFrame)
    geometry: pd.DataFrame = field(default_factory=pd.DataFrame)
    cell_parameters: pd.DataFrame = field(default_factory=pd.DataFrame)
    energy: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        for name in ["charge", "heatfo", "geometry", "cell_parameters", "energy"]:
            if not isinstance(getattr(self, name), pd.DataFrame):
                raise TypeError(f"ForceFieldOptimizationTrainingSetData.{name} must be a pandas DataFrame.")


@dataclass
class ForceFieldOptimizationParameterData:
    """Canonical optimization-parameter definition model parsed from ``params``."""

    ff_section: np.ndarray
    ff_section_line: np.ndarray
    ff_parameter: np.ndarray
    search_interval: np.ndarray
    min_value: np.ndarray
    max_value: np.ndarray
    inline_comment: np.ndarray
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        fields = ["ff_section", "ff_section_line", "ff_parameter", "search_interval", "min_value", "max_value", "inline_comment"]
        lens = []
        for name in fields:
            arr = _as_1d(f"ForceFieldOptimizationParameterData.{name}", getattr(self, name))
            lens.append(arr.shape[0])
        if len(set(lens)) > 1:
            raise ValueError("ForceFieldOptimizationParameterData arrays must have identical length.")


@dataclass
class GeometrySummaryData:
    """Canonical structure-summary model parsed from ``fort.74``."""

    identifiers: np.ndarray
    minimum_energy: Optional[np.ndarray] = None
    iterations: Optional[np.ndarray] = None
    formation_energy: Optional[np.ndarray] = None
    volume: Optional[np.ndarray] = None
    density: Optional[np.ndarray] = None
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        n_rows = _as_1d("GeometrySummaryData.identifiers", self.identifiers).shape[0]
        for name in ["minimum_energy", "iterations", "formation_energy", "volume", "density"]:
            vals = getattr(self, name)
            if vals is None:
                continue
            arr = _as_1d(f"GeometrySummaryData.{name}", vals)
            if arr.shape[0] != n_rows:
                raise ValueError(f"GeometrySummaryData.{name} length must match identifiers length.")


@dataclass
class PartialEnergyData:
    """Canonical partial-energy time-series model parsed from ``fort.73``-style files."""

    iterations: np.ndarray
    components: tuple[str, ...] = ()
    values: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        it = _as_1d("PartialEnergyData.iterations", self.iterations, dtype=int)
        vals = _as_2d("PartialEnergyData.values", self.values, dtype=float)
        if vals.shape[0] != it.shape[0]:
            raise ValueError("PartialEnergyData.values row count must match iterations length.")
        if self.components and vals.shape[1] != len(self.components):
            raise ValueError("PartialEnergyData.values column count must match components length.")


@dataclass
class StressData:
    """Canonical per-atom stress tensor time-series model."""

    iterations: np.ndarray
    components: tuple[str, ...] = ("xx", "yy", "zz", "yx", "zx", "zy")
    values: np.ndarray = field(default_factory=lambda: np.empty((0, 0, 0), dtype=float))
    average_values: Optional[np.ndarray] = None
    iso_values: Optional[np.ndarray] = None
    avg_iso_values: Optional[np.ndarray] = None
    loavg_values: Optional[np.ndarray] = None
    loavg_iso_values: Optional[np.ndarray] = None
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        it = _as_1d("StressData.iterations", self.iterations, dtype=int)
        n_frames = it.shape[0]
        n_atoms: int | None = None

        def _validate_tensor(name: str, vals: Any, required: bool) -> None:
            nonlocal n_atoms
            if vals is None:
                if required:
                    raise ValueError(f"StressData.{name} is required.")
                return
            arr = np.asarray(vals, dtype=float)
            if arr.ndim != 3:
                raise ValueError(f"StressData.{name} must be 3D; got shape={arr.shape}.")
            if arr.shape[0] != n_frames:
                raise ValueError(f"StressData.{name} frame count must match iterations length.")
            if self.components and arr.shape[2] != len(self.components):
                raise ValueError(f"StressData.{name} component count must match components length.")
            if n_atoms is None:
                n_atoms = int(arr.shape[1])
            elif int(arr.shape[1]) != n_atoms:
                raise ValueError(f"StressData.{name} atom count must match other stress arrays.")

        def _validate_scalar(name: str, vals: Any) -> None:
            nonlocal n_atoms
            if vals is None:
                return
            arr = np.asarray(vals, dtype=float)
            if arr.ndim != 2:
                raise ValueError(f"StressData.{name} must be 2D; got shape={arr.shape}.")
            if arr.shape[0] != n_frames:
                raise ValueError(f"StressData.{name} frame count must match iterations length.")
            if n_atoms is None:
                n_atoms = int(arr.shape[1])
            elif int(arr.shape[1]) != n_atoms:
                raise ValueError(f"StressData.{name} atom count must match other stress arrays.")

        _validate_tensor("values", self.values, required=True)
        _validate_tensor("average_values", self.average_values, required=False)
        _validate_tensor("loavg_values", self.loavg_values, required=False)
        _validate_scalar("iso_values", self.iso_values)
        _validate_scalar("avg_iso_values", self.avg_iso_values)
        _validate_scalar("loavg_iso_values", self.loavg_iso_values)


@dataclass
class AtomTemperatureData:
    """Canonical per-atom temperature time-series model."""

    iterations: np.ndarray
    temperatures: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        it = _as_1d("AtomTemperatureData.iterations", self.iterations, dtype=int)
        temps = _as_2d("AtomTemperatureData.temperatures", self.temperatures, dtype=float)
        if temps.shape[0] != it.shape[0]:
            raise ValueError("AtomTemperatureData.temperatures row count must match iterations length.")


@dataclass
class AtomStrainEnergyData:
    """Canonical per-atom strain-energy time-series model."""

    iterations: np.ndarray
    strain_energy: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        it = _as_1d("AtomStrainEnergyData.iterations", self.iterations, dtype=int)
        vals = _as_2d("AtomStrainEnergyData.strain_energy", self.strain_energy, dtype=float)
        if vals.shape[0] != it.shape[0]:
            raise ValueError("AtomStrainEnergyData.strain_energy row count must match iterations length.")


@dataclass
class RestraintData:
    """Canonical restraint-monitor model parsed from ``fort.76``."""

    iterations: np.ndarray
    restraint_energy: Optional[np.ndarray] = None
    potential_energy: Optional[np.ndarray] = None
    target_values: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    actual_values: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        it = _as_1d("RestraintData.iterations", self.iterations, dtype=int)
        n_rows = it.shape[0]
        if self.restraint_energy is not None:
            _require_len("RestraintData.restraint_energy", _as_1d("RestraintData.restraint_energy", self.restraint_energy), n_rows)
        if self.potential_energy is not None:
            _require_len("RestraintData.potential_energy", _as_1d("RestraintData.potential_energy", self.potential_energy), n_rows)
        tv = _as_2d("RestraintData.target_values", self.target_values, dtype=float)
        av = _as_2d("RestraintData.actual_values", self.actual_values, dtype=float)
        if tv.shape[0] != n_rows or av.shape[0] != n_rows:
            raise ValueError("RestraintData target/actual row count must match iterations length.")


@dataclass
class GeometryOptimizationProgressData:
    """Canonical geometry-optimization summary model parsed from ``fort.57``."""

    optimization_iterations: np.ndarray
    potential_energy: Optional[np.ndarray] = None
    temperature: Optional[np.ndarray] = None
    temperature_setpoint: Optional[np.ndarray] = None
    rms_gradient: Optional[np.ndarray] = None
    n_force_calls: Optional[np.ndarray] = None
    geo_descriptor: str = ""
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        it = _as_1d("GeometryOptimizationProgressData.optimization_iterations", self.optimization_iterations, dtype=int)
        n_rows = it.shape[0]
        for name in ["potential_energy", "temperature", "temperature_setpoint", "rms_gradient", "n_force_calls"]:
            vals = getattr(self, name)
            if vals is None:
                continue
            arr = _as_1d(f"GeometryOptimizationProgressData.{name}", vals)
            if arr.shape[0] != n_rows:
                raise ValueError(f"GeometryOptimizationProgressData.{name} length must match optimization_iterations.")


@dataclass
class ControlParametersData:
    """Engine-agnostic control-parameter model grouped by normalized sections."""

    general: dict[str, Any] = field(default_factory=dict)
    md: dict[str, Any] = field(default_factory=dict)
    mm: dict[str, Any] = field(default_factory=dict)
    ff: dict[str, Any] = field(default_factory=dict)
    outdated: dict[str, Any] = field(default_factory=dict)
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        for name in ["general", "md", "mm", "ff", "outdated"]:
            if not isinstance(getattr(self, name), dict):
                raise TypeError(f"ControlParametersData.{name} must be a dict.")


@dataclass
class ChargeData:
    """Canonical per-atom charge model."""

    charges: np.ndarray  # (n_frames, n_atoms_max), padded with NaN when an atom is absent
    total_charge: Optional[np.ndarray] = None  # (n_frames,)
    simulation: Optional[SimulationData] = None
    iterations: Optional[np.ndarray] = None
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        charges = _as_2d("ChargeData.charges", self.charges, dtype=float)
        n_frames, n_atoms = charges.shape
        if self.total_charge is not None:
            arr = _as_1d("ChargeData.total_charge", self.total_charge, dtype=float)
            if arr.shape[0] != n_frames:
                raise ValueError("ChargeData.total_charge length must match frame count.")
        if self.iterations is not None:
            arr = _as_1d("ChargeData.iterations", self.iterations, dtype=int)
            if arr.shape[0] != n_frames:
                raise ValueError("ChargeData.iterations length must match frame count.")
        if self.simulation is not None:
            self.simulation.validate()
            if len(self.simulation.atom_ids) != n_atoms:
                raise ValueError("ChargeData.simulation.atom_ids length must match atom count.")


@dataclass
class ElectricFieldData:
    """Canonical electric-field model."""

    applied_field_values: np.ndarray
    applied_field_components: Sequence[str] = ()
    field_energy_values: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    field_energy_components: Sequence[str] = ()
    sampled_field_iterations: Optional[np.ndarray] = None
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        applied = np.asarray(self.applied_field_values, dtype=float)
        energy = np.asarray(self.field_energy_values, dtype=float)
        if applied.ndim not in (1, 2):
            raise ValueError("ElectricFieldData.applied_field_values must be 1D or 2D.")
        if energy.ndim not in (1, 2):
            raise ValueError("ElectricFieldData.field_energy_values must be 1D or 2D.")

        if applied.ndim == 1 and self.applied_field_components and len(self.applied_field_components) != 1:
            raise ValueError("ElectricFieldData.applied_field_components must have one entry for 1D values.")
        if applied.ndim == 2 and self.applied_field_components and len(self.applied_field_components) != applied.shape[1]:
            raise ValueError("ElectricFieldData.applied_field_components length must match value columns.")
        if energy.ndim == 1 and self.field_energy_components and len(self.field_energy_components) != 1:
            raise ValueError("ElectricFieldData.field_energy_components must have one entry for 1D values.")
        if energy.ndim == 2 and self.field_energy_components and len(self.field_energy_components) != energy.shape[1]:
            raise ValueError("ElectricFieldData.field_energy_components length must match value columns.")

        n_samples = 0
        if applied.size:
            n_samples = applied.shape[0]
        if energy.size:
            if n_samples == 0:
                n_samples = energy.shape[0]
            elif n_samples != energy.shape[0]:
                raise ValueError("ElectricFieldData applied and energy values must have matching sample counts.")
        if self.sampled_field_iterations is not None:
            arr = _as_1d("ElectricFieldData.sampled_field_iterations", self.sampled_field_iterations, dtype=int)
            if n_samples and arr.shape[0] != n_samples:
                raise ValueError("ElectricFieldData.sampled_field_iterations length must match sample count.")


@dataclass
class AtomicKinematicsData:
    """Canonical single-snapshot atomic kinematics model parsed from ``vels``-style files."""

    coordinates: pd.DataFrame = field(default_factory=pd.DataFrame)
    velocities: pd.DataFrame = field(default_factory=pd.DataFrame)
    accelerations: pd.DataFrame = field(default_factory=pd.DataFrame)
    previous_accelerations: pd.DataFrame = field(default_factory=pd.DataFrame)
    lattice_parameters: Optional[dict[str, float]] = None
    md_temperature_K: Optional[float] = None
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        for name in ["coordinates", "velocities", "accelerations", "previous_accelerations"]:
            if not isinstance(getattr(self, name), pd.DataFrame):
                raise TypeError(f"AtomicKinematicsData.{name} must be a pandas DataFrame.")


@dataclass
class EregimeData:
    """Canonical electric-field schedule model parsed from ``eregime.in``."""

    iterations: np.ndarray
    field_zones: np.ndarray
    field_dir: np.ndarray
    field: np.ndarray
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        it = _as_1d("EregimeData.iterations", self.iterations, dtype=int)
        zones = _as_1d("EregimeData.field_zones", self.field_zones)
        dirs = _as_1d("EregimeData.field_dir", self.field_dir)
        fld = _as_1d("EregimeData.field", self.field)
        n = it.shape[0]
        if zones.shape[0] != n or dirs.shape[0] != n or fld.shape[0] != n:
            raise ValueError("EregimeData arrays must have identical lengths.")


@dataclass
class MolecularAnalysisData:
    """Canonical molecular-fragment analysis model."""

    iterations: np.ndarray
    totals: pd.DataFrame
    molecular_species: pd.DataFrame

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        _as_1d("MolecularAnalysisData.iterations", self.iterations, dtype=int)
        if not isinstance(self.totals, pd.DataFrame):
            raise TypeError("MolecularAnalysisData.totals must be a pandas DataFrame.")
        if not isinstance(self.molecular_species, pd.DataFrame):
            raise TypeError("MolecularAnalysisData.molecular_species must be a pandas DataFrame.")


# ---------------------------------------------------------------------
# Composite Data Models
# ---------------------------------------------------------------------

@dataclass
class ForceFieldOptimizationData:
    """Composite force-field optimization view spanning progress and diagnostics."""

    force_field_parameters: Optional[ForceFieldParametersData] = None
    optimization_parameters: Optional[ForceFieldOptimizationParameterData] = None
    training_set: Optional[ForceFieldOptimizationTrainingSetData] = None
    progress: Optional[ForceFieldOptimizationProgressData] = None
    diagnostics: Optional[ForceFieldOptimizationDiagnosticData] = None
    report: Optional[ForceFieldOptimizationReportData] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class ForceFieldOptimizationParameterBundleData:
    """Composite data for parameter optimization tasks.

    Contains exactly the two sources needed by parameter tasks:
    - optimization_parameters: parsed ``params`` data (always required)
    - force_field_parameters: parsed ``ffield`` data (used when interpretation is requested)
    """

    optimization_parameters: ForceFieldOptimizationParameterData
    force_field_parameters: Optional[ForceFieldParametersData] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class ForceFieldOptimizationDiagnosticBundleData:
    """Composite data for diagnostic optimization tasks.

    - diagnostics: parsed diagnostic data (required)
    - force_field_parameters: parsed force-field parameter tables (required for identifier interpretation)
    """

    diagnostics: ForceFieldOptimizationDiagnosticData
    force_field_parameters: ForceFieldParametersData
    metadata: Optional[dict[str, Any]] = None


@dataclass
class ForceFieldOptimizationReportEOSBundleData:
    """Composite data for EOS report tasks.

    - report: optimization report data (required)
    - geometry_summary: structure-summary data with volume mapping (required)
    """

    report: ForceFieldOptimizationReportData
    geometry_summary: GeometrySummaryData
    metadata: Optional[dict[str, Any]] = None


@dataclass
class CoordinationStatusBundleData:
    """Composite data for coordination-status analysis tasks.

    - connectivity: bond-order/neighbor information per frame (required)
    - force_field_parameters: force-field atom parameters with ``valency`` column (required)
    """

    connectivity: ConnectivityData
    force_field_parameters: ForceFieldParametersData
    metadata: Optional[dict[str, Any]] = None


@dataclass
class ElectrostaticsData:
    """Composite electrostatics model for engine-agnostic analyses."""

    trajectory: TrajectoryData
    charges: ChargeData
    connectivity: Optional[ConnectivityData] = None
    electric_field: Optional[ElectricFieldData] = None


@dataclass
class ConnectivityTrajectoryData:
    """Composite connectivity + trajectory model for coupled workflows."""

    connectivity: ConnectivityData
    trajectory: TrajectoryData
    force_field_parameters: Optional[ForceFieldParametersData] = None


