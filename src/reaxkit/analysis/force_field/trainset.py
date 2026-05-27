"""Force-field training-set analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any

import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ForceFieldOptimizationTrainingSetData
from reaxkit.presentation.specs import PresentationSpec

_TRAINSET_SECTION_ALIASES = {
    "all": "all",
    "charge": "CHARGE",
    "heatfo": "HEATFO",
    "geometry": "GEOMETRY",
    "cell": "CELL_PARAMETERS",
    "cell_parameters": "CELL_PARAMETERS",
    "cell parameters": "CELL_PARAMETERS",
    "energy": "ENERGY",
}


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
) -> pd.DataFrame:
    tables = _get_trainset_section_tables(data)
    rows: list[dict[str, str]] = []
    for section_name, df in tables.items():
        if "group_comment" not in df.columns:
            continue
        work = df.copy()
        work["group_comment"] = work["group_comment"].astype(str).str.strip()
        work = work[work["group_comment"] != ""]
        if work.empty:
            continue
        if "line_number" in work.columns:
            work["line_number"] = pd.to_numeric(work["line_number"], errors="coerce")
            reduced = (
                work.groupby("group_comment", as_index=False)["line_number"]
                .min()
                .sort_values("line_number", kind="stable", na_position="last")
            )
            for _, row in reduced.iterrows():
                rows.append(
                    {
                        "section": section_name.lower(),
                        "group_comment": str(row["group_comment"]),
                        "line_number": row["line_number"],
                        "count": 1,
                    }
                )
        else:
            for group_comment in work["group_comment"].unique():
                rows.append(
                    {
                        "section": section_name.lower(),
                        "group_comment": str(group_comment),
                        "line_number": pd.NA,
                        "count": 1,
                    }
                )

    out = pd.DataFrame(rows).drop_duplicates(ignore_index=True)
    if out.empty:
        return out
    out["line_number"] = pd.to_numeric(out["line_number"], errors="coerce")
    return out.sort_values(["line_number"], kind="stable", na_position="last").reset_index(drop=True)


def _normalize_trainset_section(section: str) -> str:
    key = str(section).strip().lower().replace("-", "_")
    if key not in _TRAINSET_SECTION_ALIASES:
        raise KeyError(
            f"Unknown trainset section {section!r}. Valid options: "
            f"{sorted(_TRAINSET_SECTION_ALIASES)}"
        )
    return _TRAINSET_SECTION_ALIASES[key]


def _build_trainset_data_table(
    data: ForceFieldOptimizationTrainingSetData,
    section: str,
) -> pd.DataFrame:
    tables = _get_trainset_section_tables(data)
    section_key = _normalize_trainset_section(section)
    if section_key == "all":
        frames: list[pd.DataFrame] = []
        for sec_name, sec_df in tables.items():
            work = sec_df.copy()
            section_value = sec_name.lower()
            if "section" in work.columns:
                work["section"] = section_value
                ordered_cols = ["section"] + [c for c in work.columns if c != "section"]
                work = work.loc[:, ordered_cols]
            else:
                work.insert(0, "section", section_value)
            frames.append(work)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return tables[section_key].copy().reset_index(drop=True)


@dataclass
class GetTrainsetDataRequest(BaseRequest):
    """Request for trainset row extraction.

    Parameters
    ----------
    section
        Trainset section to extract.

        Examples
        --------
        - ``"all"``: concatenate all sections into one table and prepend a
          ``section`` column.
        - ``"energy"``: return only ENERGY rows.
        - ``"cell_parameters"``: return only CELL PARAMETERS rows.
    """

    section: str = dc_field(
        default="all",
        metadata={
            "label": "Section",
            "help": (
                "Trainset section to return. "
                "Use 'all' to merge all sections with a leading 'section' column. "
                "Examples: 'energy', 'geometry', 'cell_parameters'."
            ),
            "choices": ["all", "charge", "heatfo", "geometry", "cell_parameters", "energy"],
        },
    )


@dataclass
class GetTrainsetDataResult(BaseResult):
    """Trainset data extraction result.

    Output structure
    ----------------
    - ``request``: the :class:`GetTrainsetDataRequest` used to generate this result.
    - ``table``: pandas.DataFrame with trainset rows for the requested section.

    Returned table behavior
    -----------------------
    - If ``request.section == "all"``, the table contains concatenated rows from
      CHARGE, HEATFO, GEOMETRY, CELL_PARAMETERS, and ENERGY, with a leading
      ``section`` column (lowercase section name).
    - For a single section request (for example ``"energy"``), the table
      contains only that section's native columns (such as ``line_number``,
      ``weight``, ``lit``, etc.).

    Example
    -------
    A single-row ENERGY output may include:
    ``section='ENERGY', line_number=142, op1='+', id1='bulk_1', n1=1.0, lit=-15.4``.
    """

    table: pd.DataFrame
    request: GetTrainsetDataRequest


@register_task("trainset_data", label="Trainset Data")
class GetTrainsetDataTask(AnalysisTask):
    """Return trainset rows for one section or all sections."""

    required_data = ForceFieldOptimizationTrainingSetData

    @staticmethod
    def recommended_presentations(
        _result: GetTrainsetDataResult,
        payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        rows = payload.get("table")
        if not isinstance(rows, list) or not rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]

        sample = rows[0] if isinstance(rows[0], dict) else {}
        if not isinstance(sample, dict):
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]

        numeric_cols = [k for k, v in sample.items() if isinstance(v, (int, float))]
        x_col = "line_number" if "line_number" in sample else (numeric_cols[0] if numeric_cols else "line_number")

        if "lit" in sample and "lit" != x_col:
            y_col = "lit"
        else:
            y_col = next((c for c in numeric_cols if c != x_col), "lit")

        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label=f"{y_col} vs {x_col}",
                mapping={"x_col": x_col, "y_col": y_col, "group_by_col": "section" if "section" in sample else ""},
                options={
                    "title": f"Trainset Data: {y_col} vs {x_col}",
                    "xlabel": x_col,
                    "ylabel": y_col,
                    "legend": "section" in sample,
                },
                view_type="plot2d",
            ),
        ]

    def run(
        self,
        data: ForceFieldOptimizationTrainingSetData,
        request: GetTrainsetDataRequest,
        reporter=None,
    ) -> GetTrainsetDataResult:
        table = _build_trainset_data_table(data, section=request.section)
        return GetTrainsetDataResult(table=table, request=request)


@dataclass
class TrainsetGroupCommentsRequest(BaseRequest):
    """Request for unique trainset group comments."""

    section: str = dc_field(
        default="all",
        metadata={
            "label": "Section",
            "help": (
                "Trainset section to return comments for. "
                "Use 'all' for all sections. "
                "Examples: 'energy', 'geometry', 'cell_parameters'."
            ),
            "choices": ["all", "charge", "heatfo", "geometry", "cell_parameters", "energy"],
        },
    )


@dataclass
class TrainsetGroupCommentsResult(BaseResult):
    """Trainset group-comment extraction result.

    Output structure:
    - request: TrainsetGroupCommentsRequest used to generate this result.
    - table: pandas.DataFrame with columns:
      ['section', 'group_comment', 'line_number', 'count']
      - section: lower-case training-set section name (for example 'energy', 'charge')
      - group_comment: unique comment text in that section
      - line_number: source line number of the first matching entry in the training-set file
      - count: per-row count placeholder (always 1); useful for counting comments per section in plots

    Example:
    ('energy', 'equation_of_state_reference_set') indicates this comment
    appeared in the ENERGY section.
    """

    table: pd.DataFrame
    request: TrainsetGroupCommentsRequest


@register_task("trainset_group_comments", label="Trainset Group Comments")
class TrainsetGroupCommentsTask(AnalysisTask):
    """Return unique trainset group-comment annotations by section."""

    required_data = ForceFieldOptimizationTrainingSetData

    @staticmethod
    def recommended_presentations(
        _result: TrainsetGroupCommentsResult, payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        rows = payload.get("table")
        if not isinstance(rows, list) or not rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = rows[0] if isinstance(rows[0], dict) else {}
        if "section" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="Comment Count by Section",
                mapping={"x_col": "section", "y_col": "count", "group_by_col": ""},
                options={
                    "title": "Comment Count by Section",
                    "xlabel": "section",
                    "ylabel": "count",
                    "legend": False,
                },
                view_type="plot2d",
            ),
        ]

    def run(
        self,
        data: ForceFieldOptimizationTrainingSetData,
        request: TrainsetGroupCommentsRequest,
        reporter=None,
    ) -> TrainsetGroupCommentsResult:
        table = _get_trainset_group_comments(data)
        section_key = _normalize_trainset_section(request.section)
        if section_key != "all" and not table.empty and "section" in table.columns:
            table = table.loc[table["section"] == section_key.lower()].copy().reset_index(drop=True)
        return TrainsetGroupCommentsResult(table=table, request=request)


__all__ = [
    "GetTrainsetDataRequest",
    "GetTrainsetDataResult",
    "GetTrainsetDataTask",
    "TrainsetGroupCommentsRequest",
    "TrainsetGroupCommentsResult",
    "TrainsetGroupCommentsTask",
]
