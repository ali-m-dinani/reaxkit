"""Engine-agnostic domain data models for analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd


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
    atom_type_nums: Optional[np.ndarray] = None   # (n_frames, n_atoms)
    molecule_nums: Optional[np.ndarray] = None    # (n_frames, n_atoms)
    cell_lengths: Optional[np.ndarray] = None  # (n_frames, 3): a, b, c
    cell_angles: Optional[np.ndarray] = None   # (n_frames, 3): alpha, beta, gamma


@dataclass
class TrajectoryData:
    """Canonical trajectory model consumed by analysis tasks."""

    positions: np.ndarray  # (n_frames, n_atoms, 3)
    elements: list[str]
    atom_ids: list[int]
    simulation: Optional[SimulationData] = None
    iterations: Optional[np.ndarray] = None
    atom_labels: Optional[np.ndarray] = None  # (n_frames, n_atoms) optional per-frame output labels


@dataclass
class GeometryData:
    """Canonical single-structure geometry model."""

    coordinates: pd.DataFrame = field(default_factory=pd.DataFrame)
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
    sum_bond_orders: Optional[np.ndarray] = None  # (n_frames, n_atoms)
    num_lone_pairs: Optional[np.ndarray] = None  # (n_frames, n_atoms)
    num_of_bonds: Optional[np.ndarray] = None    # (n_frames,)
    total_bond_order: Optional[np.ndarray] = None
    total_lone_pairs: Optional[np.ndarray] = None
    total_bond_order_uncorrected: Optional[np.ndarray] = None
    atom_ids: Optional[Sequence[int]] = None
    elements: Optional[list[str]] = None
    simulation: Optional[SimulationData] = None
    iterations: Optional[np.ndarray] = None
    metadata: Optional[dict[str, Any]] = None


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


@dataclass
class ForceFieldOptimizationProgressData:
    """Canonical force-field optimization error model parsed from ``fort.13``."""

    epochs: np.ndarray
    total_ff_error: np.ndarray
    metadata: Optional[dict[str, Any]] = None


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


@dataclass
class PartialEnergyData:
    """Canonical partial-energy time-series model parsed from ``fort.73``-style files."""

    iterations: np.ndarray
    components: tuple[str, ...] = ()
    values: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    metadata: Optional[dict[str, Any]] = None


@dataclass
class RestraintData:
    """Canonical restraint-monitor model parsed from ``fort.76``."""

    iterations: np.ndarray
    restraint_energy: Optional[np.ndarray] = None
    potential_energy: Optional[np.ndarray] = None
    target_values: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    actual_values: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    metadata: Optional[dict[str, Any]] = None


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


@dataclass
class ControlParametersData:
    """Engine-agnostic control-parameter model grouped by normalized sections."""

    general: dict[str, Any] = field(default_factory=dict)
    md: dict[str, Any] = field(default_factory=dict)
    mm: dict[str, Any] = field(default_factory=dict)
    ff: dict[str, Any] = field(default_factory=dict)
    outdated: dict[str, Any] = field(default_factory=dict)
    metadata: Optional[dict[str, Any]] = None


@dataclass
class ChargeData:
    """Canonical per-atom charge model."""

    charges: np.ndarray  # (n_frames, n_atoms)
    total_charge: Optional[np.ndarray] = None  # (n_frames,)
    simulation: Optional[SimulationData] = None
    iterations: Optional[np.ndarray] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class ElectricFieldData:
    """Canonical electric-field model."""

    applied_field_values: np.ndarray
    applied_field_components: Sequence[str] = ()
    field_energy_values: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    field_energy_components: Sequence[str] = ()
    sampled_field_iterations: Optional[np.ndarray] = None
    metadata: Optional[dict[str, Any]] = None


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


@dataclass
class EregimeData:
    """Canonical electric-field schedule model parsed from ``eregime.in``."""

    iterations: np.ndarray
    field_zones: np.ndarray
    field_dir: np.ndarray
    field: np.ndarray
    metadata: Optional[dict[str, Any]] = None


@dataclass
class MolecularAnalysisData:
    """Canonical molecular-fragment analysis model."""

    iterations: np.ndarray
    totals: pd.DataFrame
    molecular_species: pd.DataFrame


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


