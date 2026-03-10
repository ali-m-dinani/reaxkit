"""Force-field optimization analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Optional, Sequence

import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ForceFieldOptimizationProgressData
from reaxkit.presentation.specs import PresentationSpec


def _optimization_progress_table(
    data: ForceFieldOptimizationProgressData,
    epochs: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Extract optimization error progression for all or selected epochs."""
    df = pd.DataFrame(
        {
            "epoch": pd.Series(data.epochs, dtype=int),
            "total_ff_error": pd.Series(data.total_ff_error, dtype=float),
        }
    )
    if epochs is not None:
        chosen = {int(epoch) for epoch in epochs}
        df = df[df["epoch"].isin(chosen)].reset_index(drop=True)
    return df[["epoch", "total_ff_error"]].copy().sort_values("epoch").reset_index(drop=True)


@dataclass
class ForceFieldOptimizationRequest(BaseRequest):
    """Request for force-field optimization progress data."""

    epochs: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={
            "label": "Epochs",
            "help": (
                "Optional optimization epochs to include. "
                "Example: [1, 5, 10]. If omitted, all available epochs are returned."
            ),
        },
    )


@dataclass
class ForceFieldOptimizationResult(BaseResult):
    """Force-field optimization progress result.

    Output structure:
    - request: ForceFieldOptimizationRequest used to generate this result.
    - table: pandas.DataFrame with columns:
      ['epoch', 'total_ff_error']
      - epoch: optimization step index.
      - total_ff_error: total model error at that epoch.

    Example:
    - epoch=1, total_ff_error=15324.4
    - epoch=2, total_ff_error=14980.1
    """

    table: pd.DataFrame
    request: ForceFieldOptimizationRequest


@register_task("force_field_optimization")
class ForceFieldOptimizationTask(AnalysisTask):
    """Return total force-field error versus optimization epoch."""

    required_data = ForceFieldOptimizationProgressData

    @staticmethod
    def recommended_presentations(
        _result: ForceFieldOptimizationResult, payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        rows = payload.get("table")
        if not isinstance(rows, list) or not rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = rows[0] if isinstance(rows[0], dict) else {}
        if "epoch" not in sample or "total_ff_error" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="Total FF Error vs Epoch",
                mapping={"x_col": "epoch", "y_col": "total_ff_error", "group_by_col": ""},
                options={
                    "title": "Total FF Error vs Epoch",
                    "xlabel": "epoch",
                    "ylabel": "total_ff_error",
                    "legend": False,
                },
                view_type="plot2d",
            ),
        ]

    def run(
        self,
        data: ForceFieldOptimizationProgressData,
        request: ForceFieldOptimizationRequest,
        reporter=None,
    ) -> ForceFieldOptimizationResult:
        table = _optimization_progress_table(data, epochs=request.epochs)
        return ForceFieldOptimizationResult(table=table, request=request)


__all__ = [
    "ForceFieldOptimizationRequest",
    "ForceFieldOptimizationResult",
    "ForceFieldOptimizationTask",
]
