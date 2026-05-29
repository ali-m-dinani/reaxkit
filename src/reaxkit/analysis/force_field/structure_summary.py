"""Extract structure-summary analyzer tables from geometry summary bundles.

This module converts parsed geometry-summary records into typed, sortable
tabular outputs used in force-field analysis workflows. It is bounded to
summary extraction and does not perform additional structural computations.

**Usage context**

- Structure QA: Review minima, iteration counts, and summary diagnostics.
- Training artifacts: Inspect geometry-summary data used in fitting cycles.
- Report pipelines: Export structure-summary tables for post-processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import GeometrySummaryData
from reaxkit.presentation.specs import PresentationSpec


def _fort74_frame(data: GeometrySummaryData) -> pd.DataFrame:
    """Build the base structure-summary table from geometry summary fields."""
    n_rows = len(data.identifiers)
    return pd.DataFrame(
        {
            "identifier": pd.Series(data.identifiers, dtype=object),
            "Emin": (
                pd.Series(data.minimum_energy, dtype=float)
                if data.minimum_energy is not None
                else pd.Series([pd.NA] * n_rows)
            ),
            "iteration": (
                pd.Series(data.iterations, dtype=float)
                if data.iterations is not None
                else pd.Series([pd.NA] * n_rows)
            ),
            "enthalpy": (
                pd.Series(data.formation_energy, dtype=float)
                if data.formation_energy is not None
                else pd.Series([pd.NA] * n_rows)
            ),
            "volume": (
                pd.Series(data.volume, dtype=float)
                if data.volume is not None
                else pd.Series([pd.NA] * n_rows)
            ),
            "density": (
                pd.Series(data.density, dtype=float)
                if data.density is not None
                else pd.Series([pd.NA] * n_rows)
            ),
        }
    )


def _get_fort74_data(*, data: GeometrySummaryData) -> pd.DataFrame:
    """Retrieve thermodynamic summary data from structure-summary data."""
    return _fort74_frame(data)


@dataclass
class StructureSummaryRequest(BaseRequest):
    """Request payload for structure-summary extraction.

    This request configures the structure-summary analyzer task. The task
    currently exposes no request-time filters and always returns the full
    normalized structure-summary table for the loaded geometry bundle.

    Fields
    -----
    None.

    Examples
    -----
    ```python
    request = StructureSummaryRequest()
    ```
    The request asks for the full available structure-summary table.
    """


@dataclass
class StructureSummaryResult(BaseResult):
    """Result payload containing normalized structure-summary records.

    The analyzer returns one tabular view over parsed geometry summary entries,
    preserving identifiers and thermodynamic descriptors used downstream in
    force-field reporting workflows.

    Fields
    -----
    request : StructureSummaryRequest
        Request object used to generate this result.
    table : pandas.DataFrame
        Table with columns ``identifier``, ``Emin``, ``iteration``,
        ``enthalpy``, ``volume``, and ``density``.

    Examples
    -----
    ```python
    row = {
        "identifier": "bulk_0",
        "Emin": -243.1,
        "iteration": 90,
        "enthalpy": -12.8,
        "volume": 11.2,
        "density": 2.45,
    }
    ```
    The sample row represents one structure with energy and state descriptors.
    """

    table: pd.DataFrame
    request: StructureSummaryRequest


@register_task("structure_summary_data", label="Structure Summary Data")
class StructureSummaryTask(AnalysisTask):
    """Return structure-summary data."""

    required_data = GeometrySummaryData

    @staticmethod
    def recommended_presentations(
        _result: StructureSummaryResult, payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        """Suggest table and plot views for structure-summary results.

        Chooses a table view by default and adds a simple line/scatter-style
        plot when suitable ``x``/``y`` fields are present in serialized rows.

        Works on
        Analyzer task output for ``structure_summary_data``.

        Parameters
        -----
        _result : StructureSummaryResult
            Typed analyzer result instance (unused for current selection logic).
        payload : dict[str, Any]
            Serialized analyzer payload expected to include a ``table`` key.

        Returns
        -----
        list[PresentationSpec]
            Recommended presentation specs for UI renderers.

        Examples
        -----
        ```python
        specs = StructureSummaryTask.recommended_presentations(
            _result,
            {"table": [{"identifier": "bulk_0", "Emin": -243.1}]},
        )
        ```
        The returned list includes at least a table view and may include one plot.
        """
        rows = payload.get("table")
        if not isinstance(rows, list) or not rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = rows[0] if isinstance(rows[0], dict) else {}
        y_col = "Emin" if "Emin" in sample else ("enthalpy" if "enthalpy" in sample else "")
        if not y_col:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        x_col = "identifier" if "identifier" in sample else "iteration"
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label=f"{y_col} vs {x_col}",
                mapping={"x_col": x_col, "y_col": y_col, "group_by_col": ""},
                options={
                    "title": f"{y_col} vs {x_col}",
                    "xlabel": x_col,
                    "ylabel": y_col,
                    "legend": False,
                },
                view_type="plot2d",
            ),
        ]

    def run(
        self,
        data: GeometrySummaryData,
        request: StructureSummaryRequest,
        reporter=None,
    ) -> StructureSummaryResult:
        """Execute structure-summary extraction for the provided geometry data.

        Builds a normalized DataFrame from parsed geometry summary fields and
        wraps it in a typed analyzer result object.

        Works on
        ``GeometrySummaryData`` parsed from geometry summary artifacts.

        Parameters
        -----
        data : GeometrySummaryData
            Parsed geometry summary model with identifiers and optional metrics.
        request : StructureSummaryRequest
            Analyzer request configuration.
        reporter : Any, optional
            Progress reporter accepted by analyzer tasks; unused here.

        Returns
        -----
        StructureSummaryResult
            Result containing the normalized structure-summary table.

        Examples
        -----
        ```python
        result = StructureSummaryTask().run(data, StructureSummaryRequest())
        table = result.table
        ```
        ``table`` contains one row per structure identifier.
        """
        table = _get_fort74_data(
            data=data,
        )
        return StructureSummaryResult(table=table, request=request)


__all__ = [
    "StructureSummaryRequest",
    "StructureSummaryResult",
    "StructureSummaryTask",
]
