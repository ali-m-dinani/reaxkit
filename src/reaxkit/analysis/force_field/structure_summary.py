"""Structure-summary data extraction tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import GeometrySummaryData
from reaxkit.presentation.specs import PresentationSpec


def _fort74_frame(data: GeometrySummaryData) -> pd.DataFrame:
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
    """Request for structure-summary data."""


@dataclass
class StructureSummaryResult(BaseResult):
    """Structure-summary analysis result.

    Output structure:
    - request: StructureSummaryRequest used to generate this result.
    - table: pandas.DataFrame with columns:
      ['identifier', 'Emin', 'iteration', 'enthalpy', 'volume', 'density']
      - identifier: structure label
      - Emin: minimum energy
      - iteration: iteration/step index if available
      - enthalpy: formation energy
      - volume: volume
      - density: density

    Example:
    A row like ('bulk_0', -243.1, 90, -12.8, 11.2, 2.45) is one
    structure-summary record with energy and thermodynamic descriptors.
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
        table = _get_fort74_data(
            data=data,
        )
        return StructureSummaryResult(table=table, request=request)


__all__ = [
    "StructureSummaryRequest",
    "StructureSummaryResult",
    "StructureSummaryTask",
]
