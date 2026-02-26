"""ReaxFF engine adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reaxkit.core.engine_registry import register_engine
from reaxkit.domain.data_models import ConnectivityData, TrajectoryData
from reaxkit.engine.base import EngineAdapter
from reaxkit.engine.reaxff.io.xmolout_handler import XmoloutHandler


def trajectory_from_xmolout_handler(handler: XmoloutHandler) -> TrajectoryData:
    """Normalize an ``XmoloutHandler`` into ``TrajectoryData``."""
    n_frames = handler.n_frames()
    frames = [handler.frame(i) for i in range(n_frames)]
    positions = np.stack([f["coords"] for f in frames], axis=0)
    elements = list(frames[0]["atom_types"]) if frames else []
    atom_ids = list(range(1, len(elements) + 1))

    df = handler.dataframe()
    iterations = df["iter"].to_numpy() if "iter" in df.columns else np.arange(n_frames)
    times = df["time"].to_numpy() if "time" in df.columns else None

    return TrajectoryData(
        positions=positions,
        elements=elements,
        atom_ids=atom_ids,
        time=times,
        iterations=iterations,
    )


@register_engine("reaxff")
class ReaxFFAdapter(EngineAdapter):
    """Adapter that loads ReaxFF outputs into domain models."""

    def detect(self, path: str | Path) -> float:
        p = Path(path)
        has_xmol = (p / "xmolout").exists() or p.name == "xmolout"
        return 0.95 if has_xmol else 0.0

    def load(self, data_type, args: dict):
        if data_type is TrajectoryData:
            return self.load_trajectory(args)
        if data_type is ConnectivityData:
            return self.load_connectivity(args)
        raise ValueError(f"{self.name} cannot load data type: {data_type}")

    def load_trajectory(self, args: dict) -> TrajectoryData:
        xmol_path = args.get("xmolout") or args.get("input") or "xmolout"
        handler = XmoloutHandler(xmol_path)
        return trajectory_from_xmolout_handler(handler)

    def load_connectivity(self, args: dict) -> ConnectivityData:
        _ = args
        return ConnectivityData(adjacency=np.empty((0, 0), dtype=int))
