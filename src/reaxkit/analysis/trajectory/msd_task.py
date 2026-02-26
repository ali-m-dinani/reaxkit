"""Engine-agnostic MSD analysis task."""

from __future__ import annotations

import numpy as np

from reaxkit.core.analysis_task import AnalysisTask
from reaxkit.core.task_registry import register_task
from reaxkit.domain.data_models import MSDRequest, MSDResult, TrajectoryData


@register_task("msd")
class MSDTask(AnalysisTask):
    """Mean-squared displacement task on normalized trajectory data."""

    required_data = TrajectoryData

    def run(self, data: TrajectoryData, request: MSDRequest) -> MSDResult:
        positions = data.positions
        if positions.shape[0] < 2:
            return MSDResult(lag=np.array([], dtype=int), msd=np.array([], dtype=float))

        if request.atom_ids:
            idx = np.array([data.atom_ids.index(aid) for aid in request.atom_ids], dtype=int)
        else:
            idx = np.arange(positions.shape[1], dtype=int)

        ref = positions[0, idx, :]
        diffs = positions[1:, idx, :] - ref[None, :, :]
        sq = np.sum(diffs * diffs, axis=2)
        msd = np.mean(sq, axis=1)
        lag = np.arange(1, positions.shape[0], dtype=int)

        return MSDResult(lag=lag, msd=msd)
