"""LAMMPS engine adapter (scaffold)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reaxkit.core.engine_registry import register_engine
from reaxkit.domain.data_models import ConnectivityData, TrajectoryData
from reaxkit.engine.base import EngineAdapter


@register_engine("lammps")
class LAMMPSAdapter(EngineAdapter):
    """Adapter scaffold for LAMMPS trajectory formats."""

    def detect(self, path: str | Path) -> float:
        p = Path(path)
        names = {"dump.lammpstrj", "log.lammps"}
        if p.is_dir() and any((p / n).exists() for n in names):
            return 0.8
        if p.is_file() and p.name in names:
            return 0.8
        return 0.0

    def load(self, data_type, args: dict):
        if data_type is TrajectoryData:
            return self.load_trajectory(args)
        if data_type is ConnectivityData:
            return self.load_connectivity(args)
        raise ValueError(f"{self.name} cannot load data type: {data_type}")

    def load_trajectory(self, args: dict) -> TrajectoryData:
        _ = args
        raise RuntimeError("LAMMPS trajectory loading is not yet wired.")

    def load_connectivity(self, args: dict) -> ConnectivityData:
        _ = args
        return ConnectivityData(adjacency=np.empty((0, 0), dtype=int))
