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
    V: Optional[np.ndarray] = None
    T: Optional[np.ndarray] = None
    P: Optional[np.ndarray] = None
    D: Optional[np.ndarray] = None
    elap_time: Optional[np.ndarray] = None
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
class ForceFieldData:
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
class MolecularAnalysisData:
    """Canonical molecular-fragment analysis model."""

    iterations: np.ndarray
    totals: pd.DataFrame
    molecular_species: pd.DataFrame


# ---------------------------------------------------------------------
# Composite Data Models
# ---------------------------------------------------------------------

@dataclass
class ElectrostaticsData:
    """Composite electrostatics model for engine-agnostic analyses."""

    trajectory: TrajectoryData
    charges: ChargeData
    connectivity: Optional[ConnectivityData] = None
    electric_field: Optional[ElectricFieldData] = None


