"""Engine-agnostic MSD analysis task."""

from __future__ import annotations

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.task_registry import register_task
from reaxkit.domain.data_models import MSDRequest, MSDResult, TrajectoryData


@register_task("msd")
class MSDTask(AnalysisTask):
    """Per-atom mean-squared displacement over selected frames/dimensions."""

    required_data = TrajectoryData

    def run(self, data: TrajectoryData, request: MSDRequest) -> MSDResult:
        dims = tuple(d for d in request.dims if d in ("x", "y", "z"))
        if not dims:
            raise ValueError("dims must include at least one of 'x','y','z'")

        n_frames = data.positions.shape[0]
        if n_frames == 0:
            return MSDResult(table=pd.DataFrame(columns=["frame_index", "iter", "atom_id", "msd"]))

        frame_idx = list(range(n_frames)) if request.frames is None else [int(i) for i in request.frames]
        step = max(1, int(request.every))
        frame_idx = frame_idx[::step]
        if not frame_idx:
            return MSDResult(table=pd.DataFrame(columns=["frame_index", "iter", "atom_id", "msd"]))

        ref_frame = frame_idx[0] if request.origin == "first" else int(request.origin)
        if ref_frame not in frame_idx:
            raise ValueError("origin must be 'first' or a frame index inside the selected frames")

        # atom selection
        if request.atom_ids is not None:
            sel_idx = [data.atom_ids.index(int(aid)) for aid in request.atom_ids]
        elif request.atom_types:
            tset = {str(t) for t in request.atom_types}
            sel_idx = [j for j, t in enumerate(data.elements) if str(t) in tset]
        else:
            sel_idx = list(range(data.positions.shape[1]))

        if not sel_idx:
            return MSDResult(table=pd.DataFrame(columns=["frame_index", "iter", "atom_id", "msd"]))

        axes = {"x": 0, "y": 1, "z": 2}
        use_cols = [axes[d] for d in dims]

        sel = np.asarray(sel_idx, dtype=int)
        atom_ids = [data.atom_ids[i] for i in sel_idx]

        r0 = data.positions[ref_frame][sel[:, None], use_cols].astype(float)
        rows: list[dict] = []

        for i in frame_idx:
            coords = data.positions[i][sel[:, None], use_cols].astype(float)
            dr = coords - r0
            sq = np.sum(dr * dr, axis=1)

            iter_val = int(data.iterations[i]) if data.iterations is not None else int(i)
            for atom_id, msd_val in zip(atom_ids, sq):
                rows.append(
                    {
                        "frame_index": int(i),
                        "iter": iter_val,
                        "atom_id": int(atom_id),
                        "msd": float(msd_val),
                    }
                )

        table = pd.DataFrame(rows).sort_values(["frame_index", "atom_id"]).reset_index(drop=True)
        return MSDResult(table=table)
