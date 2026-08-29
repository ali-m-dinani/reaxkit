from __future__ import annotations

import numpy as np

from reaxkit.analysis.trajectory.frame_count import (
    FramesCountRequest,
    FramesCountTask,
)
from reaxkit.domain.data_models import TrajectoryData


def _trajectory(n_frames: int) -> TrajectoryData:
    return TrajectoryData(
        positions=np.zeros((n_frames, 2, 3), dtype=float),
        elements=["C", "H"],
        atom_ids=[1, 2],
    )


def test_frame_count_uses_trajectory_data_frame_dimension() -> None:
    result = FramesCountTask().run(_trajectory(4), FramesCountRequest())

    assert result.count == 4


def test_frame_count_supports_empty_trajectory_data() -> None:
    result = FramesCountTask().run(_trajectory(0), FramesCountRequest())

    assert result.count == 0
