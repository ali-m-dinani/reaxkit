"""Provide analyzer tasks for force-field optimization progress data.

This module extracts epoch-level optimization metrics and structured progress
tables from optimization logs. It is scoped to progress/error-series analysis
and does not generate final optimization reports.

**Usage context**

- Training monitoring: Track optimization error across epochs.
- Subset analysis: Slice progress by selected epoch ranges.
- Plotting/reporting: Feed normalized progress tables into visual summaries.
"""

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
    """Request payload for optimization-progress extraction.

    This request optionally filters the optimization progression table to a
    selected subset of epochs while preserving epoch ordering.

    Fields
    -----
    epochs : Optional[Sequence[int]]
        Optional set/list/sequence of epoch indices to include. If omitted,
        all available epochs from the parsed optimization data are returned.

    Examples
    -----
    ```python
    request = ForceFieldOptimizationRequest(epochs=[1, 5, 10])
    ```
    The request keeps only epochs 1, 5, and 10 in the output table.
    """

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
    """Result payload for optimization-progress analysis.

    The analyzer returns an epoch-indexed error trajectory suitable for trend
    inspection, convergence diagnostics, and downstream plotting.

    Fields
    -----
    request : ForceFieldOptimizationRequest
        Request object used to generate this result.
    table : pandas.DataFrame
        Table with columns ``epoch`` and ``total_ff_error``.

    Examples
    -----
    ```python
    rows = [
        {"epoch": 1, "total_ff_error": 15324.4},
        {"epoch": 2, "total_ff_error": 14980.1},
    ]
    ```
    Each row records model error at one optimization epoch.
    """

    table: pd.DataFrame
    request: ForceFieldOptimizationRequest


@register_task("force_field_optimization", label="Force Field Optimization")
class ForceFieldOptimizationTask(AnalysisTask):
    """Return total force-field error versus optimization epoch."""

    required_data = ForceFieldOptimizationProgressData

    @staticmethod
    def recommended_presentations(
        _result: ForceFieldOptimizationResult, payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        """Suggest default table/plot renderers for optimization progress output.

        Returns a tabular view for all outputs and adds an error-vs-epoch plot
        when the serialized table contains the expected numeric columns.

        Works on
        Analyzer task output for ``force_field_optimization``.

        Parameters
        -----
        _result : ForceFieldOptimizationResult
            Typed analyzer result instance (unused by current logic).
        payload : dict[str, Any]
            Serialized analyzer payload expected to include ``table`` rows.

        Returns
        -----
        list[PresentationSpec]
            Presentation specs appropriate for UI rendering.

        Examples
        -----
        ```python
        specs = ForceFieldOptimizationTask.recommended_presentations(
            _result,
            {"table": [{"epoch": 1, "total_ff_error": 15324.4}]},
        )
        ```
        The returned specs include a table and a ``total_ff_error`` vs ``epoch`` plot.
        """
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
        """Run the optimization-progress analyzer task.

        Extracts epoch/error rows from parsed optimization progress data,
        applies optional epoch filtering, and returns a typed result payload.

        Works on
        ``ForceFieldOptimizationProgressData`` parsed from optimization logs.

        Parameters
        -----
        data : ForceFieldOptimizationProgressData
            Parsed optimization progress record source.
        request : ForceFieldOptimizationRequest
            Request containing optional epoch filters.
        reporter : Any, optional
            Progress callback accepted by the task interface; unused here.

        Returns
        -----
        ForceFieldOptimizationResult
            Analyzer result with the normalized progress table.

        Examples
        -----
        ```python
        task = ForceFieldOptimizationTask()
        result = task.run(data, ForceFieldOptimizationRequest(epochs=[1, 2, 3]))
        ```
        The output table contains only the requested epochs when available.
        """
        table = _optimization_progress_table(data, epochs=request.epochs)
        return ForceFieldOptimizationResult(table=table, request=request)


__all__ = [
    "ForceFieldOptimizationRequest",
    "ForceFieldOptimizationResult",
    "ForceFieldOptimizationTask",
]
