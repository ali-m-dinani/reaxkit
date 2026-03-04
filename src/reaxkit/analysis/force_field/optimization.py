"""Force-field optimization analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ForceFieldOptimizationProgressData


def _get_fort13_data(
    data: ForceFieldOptimizationProgressData,
    epochs: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Extract total force-field error for all or selected epochs."""
    df = pd.DataFrame(
        {
            "epoch": pd.Series(data.epochs, dtype=int),
            "total_ff_error": pd.Series(data.total_ff_error, dtype=float),
        }
    )
    if epochs is not None:
        chosen = {int(epoch) for epoch in epochs}
        df = df[df["epoch"].isin(chosen)].reset_index(drop=True)
    return df[["epoch", "total_ff_error"]].copy()


def _error_series_across_epochs(data: ForceFieldOptimizationProgressData) -> list[float]:
    """Return the total force-field error values across all epochs."""
    return [float(v) for v in data.total_ff_error.tolist()]


@dataclass
class ForceFieldOptimizationRequest(BaseRequest):
    """Request for fort.13 optimization error data."""

    epochs: Optional[Sequence[int]] = None


@dataclass
class ForceFieldOptimizationResult(BaseResult):
    """Result for fort.13 optimization error data."""

    table: pd.DataFrame


@register_task("force_field_optimization")
class ForceFieldOptimizationTask(AnalysisTask):
    """Return total force-field error versus optimization epoch."""

    required_data = ForceFieldOptimizationProgressData

    def run(
        self,
        data: ForceFieldOptimizationProgressData,
        request: ForceFieldOptimizationRequest,
        reporter=None,
    ) -> ForceFieldOptimizationResult:
        table = _get_fort13_data(data, epochs=request.epochs)
        return ForceFieldOptimizationResult(table=table)


__all__ = [
    "ForceFieldOptimizationRequest",
    "ForceFieldOptimizationResult",
    "ForceFieldOptimizationTask",
]
