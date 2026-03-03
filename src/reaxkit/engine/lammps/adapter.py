"""LAMMPS engine adapter (scaffold)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reaxkit.core.engine_registry import register_engine
from reaxkit.domain.data_models import ConnectivityData, ConnectivityTrajectoryData, TrajectoryData
from reaxkit.engine.base import EngineAdapter
from reaxkit.engine.common.xyz_generator import write_xyz_trajectory


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

    def load_trajectory(self, args: dict) -> TrajectoryData:
        _ = args
        raise RuntimeError("LAMMPS trajectory loading is not yet wired.")

    def load_connectivity(self, args: dict) -> ConnectivityData:
        _ = args
        return ConnectivityData(connectivity=np.empty((0, 0), dtype=int))

    def load_connectivity_trajectory(self, args: dict) -> ConnectivityTrajectoryData:
        return ConnectivityTrajectoryData(
            connectivity=self.load_connectivity(args),
            trajectory=self.load_trajectory(args),
        )

    def write_trajectory(self, data: TrajectoryData, out_path: str | Path, args: dict | None = None):
        args = args or {}
        return write_xyz_trajectory(data, out_path, precision=int(args.get("precision", 6)))
