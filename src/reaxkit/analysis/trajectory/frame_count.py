"""Count frames in canonical trajectory data."""

from __future__ import annotations

from dataclasses import dataclass

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import TrajectoryData


@dataclass
class FramesCountRequest(BaseRequest):
    """Request payload for counting all frames in a trajectory."""


@dataclass
class FramesCountResult(BaseResult):
    """Number of frames stored along the trajectory frame dimension."""

    count: int
    request: FramesCountRequest


@register_task("get_frames_count", label="Trajectory Frame Count")
class FramesCountTask(AnalysisTask):
    """Count frames in any normalized :class:`TrajectoryData` instance."""

    required_data = TrajectoryData

    def run(
        self,
        data: TrajectoryData,
        request: FramesCountRequest,
        reporter=None,
    ) -> FramesCountResult:
        """Return the size of the frame axis in ``TrajectoryData.positions``."""
        _ = reporter
        return FramesCountResult(count=int(data.positions.shape[0]), request=request)


__all__ = ["FramesCountRequest", "FramesCountResult", "FramesCountTask"]
