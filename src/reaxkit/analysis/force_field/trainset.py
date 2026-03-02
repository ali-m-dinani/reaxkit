"""Force-field training-set analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ForceFieldOptimizationTrainingSetData


def _get_trainset_section_tables(data: ForceFieldOptimizationTrainingSetData) -> dict[str, pd.DataFrame]:
    return {
        "CHARGE": data.charge.copy(),
        "HEATFO": data.heatfo.copy(),
        "GEOMETRY": data.geometry.copy(),
        "CELL_PARAMETERS": data.cell_parameters.copy(),
        "ENERGY": data.energy.copy(),
    }


def _get_trainset_group_comments(
    data: ForceFieldOptimizationTrainingSetData,
    *,
    sort: bool = False,
) -> pd.DataFrame:
    tables = _get_trainset_section_tables(data)
    rows: list[dict[str, str]] = []
    for section_name, df in tables.items():
        if "group_comment" not in df.columns:
            continue
        series = df["group_comment"].astype(str).str.strip()
        series = series[series != ""]
        for group_comment in series.unique():
            rows.append({"section": section_name.lower(), "group_comment": group_comment})

    result = pd.DataFrame(rows).drop_duplicates()
    if sort and not result.empty:
        result = result.sort_values(["section", "group_comment"], ignore_index=True)
    return result


@dataclass
class TrainsetGroupCommentsRequest(BaseRequest):
    """Request for unique trainset group comments."""

    sort: bool = False


@dataclass
class TrainsetGroupCommentsResult(BaseResult):
    """Result for unique trainset group comments."""

    table: pd.DataFrame


@register_task("trainset_group_comments")
class TrainsetGroupCommentsTask(AnalysisTask):
    """Return unique trainset group-comment annotations by section."""

    required_data = ForceFieldOptimizationTrainingSetData

    def run(
        self,
        data: ForceFieldOptimizationTrainingSetData,
        request: TrainsetGroupCommentsRequest,
        reporter=None,
    ) -> TrainsetGroupCommentsResult:
        table = _get_trainset_group_comments(data, sort=bool(request.sort))
        return TrainsetGroupCommentsResult(table=table)


__all__ = [
    "TrainsetGroupCommentsRequest",
    "TrainsetGroupCommentsResult",
    "TrainsetGroupCommentsTask",
]
