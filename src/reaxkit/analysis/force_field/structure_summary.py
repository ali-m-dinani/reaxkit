"""Structure-summary data extraction tasks."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import GeometrySummaryData


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
            "iter": (
                pd.Series(data.iterations, dtype=float)
                if data.iterations is not None
                else pd.Series([pd.NA] * n_rows)
            ),
            "Hf": (
                pd.Series(data.formation_energy, dtype=float)
                if data.formation_energy is not None
                else pd.Series([pd.NA] * n_rows)
            ),
            "V": (
                pd.Series(data.volume, dtype=float)
                if data.volume is not None
                else pd.Series([pd.NA] * n_rows)
            ),
            "D": (
                pd.Series(data.density, dtype=float)
                if data.density is not None
                else pd.Series([pd.NA] * n_rows)
            ),
        }
    )


def _get_fort74_data(
    *,
    data: GeometrySummaryData,
    sort: str | None = None,
    ascending: bool = True,
) -> pd.DataFrame:
    """Retrieve thermodynamic summary data from ``fort.74``-style data."""
    df = _fort74_frame(data)
    if not sort:
        return df
    return df.sort_values(by=str(sort), ascending=bool(ascending)).reset_index(drop=True)


@dataclass
class StructureSummaryRequest(BaseRequest):
    """Request for fort.74 structure-summary data."""

    sort: str | None = None
    ascending: bool = True


@dataclass
class StructureSummaryResult(BaseResult):
    """Result for fort.74 structure-summary data."""

    table: pd.DataFrame


@register_task("structure_summary_data")
class StructureSummaryTask(AnalysisTask):
    """Return structure-summary data from fort.74."""

    required_data = GeometrySummaryData

    def run(
        self,
        data: GeometrySummaryData,
        request: StructureSummaryRequest,
        reporter=None,
    ) -> StructureSummaryResult:
        table = _get_fort74_data(
            data=data,
            sort=request.sort,
            ascending=bool(request.ascending),
        )
        return StructureSummaryResult(table=table)


__all__ = [
    "StructureSummaryRequest",
    "StructureSummaryResult",
    "StructureSummaryTask",
]
