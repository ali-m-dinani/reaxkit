"""Engine-agnostic domain data models for analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TrajectoryData:
    """Canonical trajectory model consumed by analysis tasks."""

    positions: np.ndarray  # (n_frames, n_atoms, 3)
    elements: list[str]
    atom_ids: list[int]
    time: Optional[np.ndarray] = None
    iterations: Optional[np.ndarray] = None


@dataclass
class ConnectivityData:
    """Canonical connectivity model."""

    adjacency: np.ndarray


@dataclass
class ChargeData:
    """Canonical per-atom charge model."""

    charges: np.ndarray
