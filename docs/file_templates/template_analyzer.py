"""Template analyzer task module using Request/Task/Result structure.

This template demonstrates the current ReaxKit analyzer pattern: one request
dataclass, one result dataclass, and one registered task class exposing
`recommended_presentations` and `run`. Replace placeholder logic, field names,
and defaults with domain-specific behavior for your analyzer.

**Usage context**

- Analyzer scaffolding: Start new analyzers with consistent task wiring.
- UI compatibility: Provide `metadata` on request fields for form rendering.
- Presentation defaults: Return table/plot specs for downstream visualization.

Notes
-----
This file is a development template and is not intended as production analysis
logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Optional, Sequence

import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import TrajectoryData
from reaxkit.presentation.specs import PresentationSpec


@dataclass
class TemplateRequest(BaseRequest):
    """Request payload template for custom analyzers.

    Carries generic selection/configuration inputs that users can copy and adapt
    for new analyzer implementations.

    Fields
    -----
    selected_ids : Optional[list[int]]
        Optional entity IDs to include (atoms, bonds, clusters, etc.).
        Default is `None` (all entities).
    selected_labels : Optional[list[str]]
        Optional label/type filters used when IDs are not provided.
        Default is `None`.
    dimensions : Sequence[str]
        Generic axis/component selector. Replace choices with domain-relevant
        values when needed. Default is `("x", "y", "z")`.
    mode : str
        Processing mode selector for branching analysis behavior.
        Default is `"default"`.
    frame_indices : Optional[Sequence[int]]
        Optional frame/sample indices to evaluate; `None` means all.
    stride : int
        Step size over selected frames/samples; must be >= 1. Default is `1`.
    include_reference : bool
        Toggle for including baseline/reference computations. Default is `True`.
    threshold : Optional[float]
        Optional numeric cutoff for filtering or event detection.
        Default is `None`.

    Examples
    -----
    Sample request payload/object:
    `TemplateRequest(selected_labels=["O"], dimensions=("x", "y"), mode="default", frame_indices=[0, 10, 20], stride=1, include_reference=True, threshold=0.2)`
    This sample configures a filtered, frame-sliced run with a numeric cutoff.
    """

    selected_ids: Optional[list[int]] = dc_field(
        default=None,
        metadata={
            "label": "Selected IDs",
            "help": "Optional entity IDs to include. Empty means all.",
            "units": "index",
        },
    )
    selected_labels: Optional[list[str]] = dc_field(
        default=None,
        metadata={
            "label": "Selected Labels",
            "help": "Optional label/type filters when selected_ids is empty.",
        },
    )
    dimensions: Sequence[str] = dc_field(
        default=("x", "y", "z"),
        metadata={
            "label": "Dimensions",
            "help": "Generic axis/component selector.",
            "choices": ["x", "y", "z"],
        },
    )
    mode: str = dc_field(
        default="default",
        metadata={
            "label": "Mode",
            "help": "Processing mode for analyzer logic branches.",
            "choices": ["default", "strict", "fast"],
        },
    )
    frame_indices: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={
            "label": "Frame Indices",
            "help": "Optional frame/sample indices. Empty means all.",
            "units": "frame_index",
        },
    )
    stride: int = dc_field(
        default=1,
        metadata={
            "label": "Stride",
            "help": "Step size over selected frames/samples.",
            "min": 1,
            "units": "frames",
        },
    )
    include_reference: bool = dc_field(
        default=True,
        metadata={
            "label": "Include Reference",
            "help": "Include baseline/reference terms in computations.",
            "choices": [True, False],
        },
    )
    threshold: Optional[float] = dc_field(
        default=None,
        metadata={
            "label": "Threshold",
            "help": "Optional numeric cutoff used by analyzer logic.",
            "min": 0.0,
        },
    )


@dataclass
class TemplateResult(BaseResult):
    """Result payload template for custom analyzers.

    Stores the produced output table and the request used to generate it so
    downstream consumers can preserve analysis provenance.

    Fields
    -----
    table : pd.DataFrame
        Output rows produced by the analyzer. Typical columns include one
        index/time axis, grouping identifiers, and one or more computed values.
    request : TemplateRequest
        Request object used for this analysis run.

    Examples
    -----
    Sample output payload/object:
    `TemplateResult(table=<DataFrame rows>, request=<TemplateRequest ...>)`
    The table carries computed rows, and `request` captures generation inputs.
    """

    table: pd.DataFrame
    request: TemplateRequest


@register_task("template_task", label="Template Task")
class TemplateTask(AnalysisTask):
    """Template analyzer task with default presentation wiring."""

    required_data = TrajectoryData

    @staticmethod
    def recommended_presentations(_result: TemplateResult, payload: dict[str, Any]) -> list[PresentationSpec]:
        """Build default presentation specs for template analyzer outputs.

        Chooses a table view and, when compatible columns exist, a simple
        2D line-plot mapping.

        Works on
        -----
        Analyzer task output payloads

        Parameters
        -----
        _result : TemplateResult
            Analysis result object for the executed task.
        payload : dict[str, Any]
            Serialized result payload used by presentation dispatch.

        Returns
        -----
        list[PresentationSpec]
            Recommended table and optional plot presentation specs.

        Examples
        -----
        ```python
        specs = TemplateTask.recommended_presentations(result, payload)
        ```
        Sample output:
        `list[PresentationSpec]` with at least a table view.
        Meaning:
        Returned specs define default ways to render this analyzer result.
        """
        rows = payload.get("table")
        if not isinstance(rows, list) or not rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = rows[0] if isinstance(rows[0], dict) else {}

        # Generic heuristic: use a time/index-like x column if present.
        x_col = ""
        for candidate in ("iter", "frame_index", "time", "x"):
            if candidate in sample:
                x_col = candidate
                break
        y_col = ""
        for candidate in ("value", "y", "metric"):
            if candidate in sample:
                y_col = candidate
                break
        if not x_col or not y_col:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]

        group_col = "group" if "group" in sample else ""
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label=f"{y_col} vs {x_col}",
                mapping={"x_col": x_col, "y_col": y_col, "group_by_col": group_col},
                options={"title": f"{y_col} vs {x_col}", "xlabel": x_col, "ylabel": y_col, "legend": bool(group_col)},
                view_type="plot2d",
            ),
        ]

    def run(self, data: TrajectoryData, request: TemplateRequest, reporter=None) -> TemplateResult:
        """Execute template analysis and return tabular results.

        This run method is scaffold logic only; replace placeholder rows with
        real computations for your analyzer.

        Works on
        -----
        TrajectoryData plus TemplateRequest analyzer inputs

        Parameters
        -----
        data : TrajectoryData
            Input data bundle for the analyzer.
        request : TemplateRequest
            Analysis configuration and selection controls.
        reporter : Any
            Optional progress callback invoked during analysis.

        Returns
        -----
        TemplateResult
            Result object containing output table and original request.

        Examples
        -----
        ```python
        task = TemplateTask()
        req = TemplateRequest(mode="default", dimensions=("x", "y", "z"))
        result = task.run(data, req)
        print(result.table.head())
        ```
        Sample output:
        A DataFrame with placeholder columns like `frame_index`, `iter`, `group`, `value`.
        Meaning:
        Each row represents one computed observation from selected samples.
        """
        _ = data
        _ = reporter
        table = pd.DataFrame(
            [
                {"frame_index": 0, "iter": 0, "group": "sample", "value": 0.0},
            ]
        )
        return TemplateResult(table=table, request=request)
