"""Engine-agnostic domain data models for analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult


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


@dataclass
class MSDRequest(BaseRequest):
    """Request for MSD analysis."""

    atom_ids: Optional[list[int]] = None
    atom_types: Optional[list[str]] = None
    dims: Sequence[str] = ("x", "y", "z")
    origin: Union[str, int] = "first"
    frames: Optional[Sequence[int]] = None
    every: int = 1


@dataclass
class MSDResult(BaseResult):
    """Result of MSD analysis."""

    table: pd.DataFrame
