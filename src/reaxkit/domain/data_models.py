"""Engine-agnostic domain data models for analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BaseRequest:
    """Base request model."""


@dataclass
class BaseResult:
    """Base result model."""


@dataclass
class TrajectoryData:
    """Canonical trajectory model consumed by analysis tasks."""

    positions: np.ndarray
    elements: list[str]
    atom_ids: list[int]
    time: Optional[np.ndarray] = None


@dataclass
class ConnectivityData:
    """Canonical connectivity model."""

    adjacency: np.ndarray


@dataclass
class ChargeData:
    """Canonical per-atom charge model."""

    charges: np.ndarray


@dataclass
class MSDRequest(BaseRequest):
    """Request for MSD analysis."""

    atom_ids: Optional[list[int]] = None


@dataclass
class MSDResult(BaseResult):
    """Result of MSD analysis."""

    lag: np.ndarray
    msd: np.ndarray
