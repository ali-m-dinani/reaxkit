"""Engine-agnostic domain data models for analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


@dataclass
class TrajectoryData:
    """Canonical trajectory model consumed by analysis tasks."""

    positions: np.ndarray  # (n_frames, n_atoms, 3)
    elements: list[str]
    atom_ids: list[int]
    time: Optional[np.ndarray] = None
    iterations: Optional[np.ndarray] = None
    cell_lengths: Optional[np.ndarray] = None  # (n_frames, 3): a, b, c
    cell_angles: Optional[np.ndarray] = None   # (n_frames, 3): alpha, beta, gamma


@dataclass
class ConnectivityData:
    """Canonical connectivity model.

    Notes
    -----
    ``adjacency`` is retained for backward compatibility, but engine-agnostic
    connectivity analyses should prefer ``bond_orders``/``sum_bond_orders``
    alongside ``atom_ids`` and ``elements``.
    """

    adjacency: Optional[np.ndarray] = None
    bond_orders: Any = None
    sum_bond_orders: Optional[np.ndarray] = None  # (n_frames, n_atoms)
    atom_ids: Optional[np.ndarray] = None
    elements: Optional[list[str]] = None
    time: Optional[np.ndarray] = None
    iterations: Optional[np.ndarray] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class AtomReferenceData:
    """Engine-agnostic atom reference properties."""

    element_props: dict[str, dict[str, Any]]
    pair_props: Optional[dict[tuple[str, str], dict[str, Any]]] = None
    global_props: Optional[dict[str, Any]] = None
    source: str = ""
    metadata: Optional[dict[str, Any]] = None


@dataclass
class ChargeData:
    """Canonical per-atom charge model."""

    charges: np.ndarray  # (n_frames, n_atoms)
    atom_ids: Optional[Sequence[int]] = None
    iterations: Optional[np.ndarray] = None
    time: Optional[np.ndarray] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class ElectricFieldData:
    """Canonical electric-field model."""

    values: np.ndarray  # (n_samples, n_components) or (n_samples,)
    components: Sequence[str] = ("field_x", "field_y", "field_z")
    iterations: Optional[np.ndarray] = None
    time: Optional[np.ndarray] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class ElectrostaticsData:
    """Composite electrostatics model for engine-agnostic analyses."""

    trajectory: TrajectoryData
    charges: ChargeData
    connectivity: Optional[ConnectivityData] = None
    electric_field: Optional[ElectricFieldData] = None


