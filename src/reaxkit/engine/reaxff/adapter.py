"""ReaxFF engine adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reaxkit.core.engine_registry import register_engine
from reaxkit.domain.data_models import ConnectivityData, TrajectoryData
from reaxkit.engine.base import EngineAdapter
from reaxkit.io.handlers.xmolout_handler import XmoloutHandler


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
        n_frames = handler.n_frames()

        frames = [handler.frame(i) for i in range(n_frames)]
        positions = np.stack([f["coords"] for f in frames], axis=0)
        elements = list(frames[0]["atom_types"]) if frames else []
        atom_ids = list(range(1, len(elements) + 1))

        times = None
        if hasattr(handler, "metadata"):
            meta = handler.metadata()
            if meta.get("has_time"):
                df = handler.dataframe()
                if "time" in df.columns:
                    times = df["time"].to_numpy()

        return TrajectoryData(positions=positions, elements=elements, atom_ids=atom_ids, time=times)

    def load_connectivity(self, args: dict) -> ConnectivityData:
        _ = args
        return ConnectivityData(adjacency=np.empty((0, 0), dtype=int))
