"""Analyze parsed force-field training-set sections as structured task outputs.

This module exposes analyzer tasks for training-set records, including
section-based extraction and comment grouping for optimization diagnostics.
It is limited to already-parsed training-set content and does not parse raw
trainset files directly.

**Usage context**

- Dataset inspection: Extract task-relevant rows from trainset sections.
- Comment analysis: Group and review training-set annotations.
- Optimization support: Supply curated trainset tables to report pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any

import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.registry.analysis_task_registry import register_task
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
    """Return per-section training-set tables keyed by canonical section names."""
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
    """Collect unique non-empty group comments across trainset sections."""
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
    """Normalize a section alias into a canonical trainset section key."""
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
    """Build a trainset table for one section or a concatenated all-section view."""
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
class TrainsetDataRequest(BaseRequest):
    """Request payload for trainset row extraction.

    This request selects a single trainset section or all supported sections
    from parsed training-set data for tabular output.

    Fields
    -----
    section : str
        Section selector. Use ``"all"`` to concatenate all sections with a
        leading ``section`` column, or one of ``"charge"``, ``"heatfo"``,
        ``"geometry"``, ``"cell_parameters"``, ``"energy"``.

    Examples
    -----
    ```python
    request = GetTrainsetDataRequest(section="energy")
    ```
    The request returns only ENERGY-section rows.
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
class TrainsetDataResult(BaseResult):
    """Result payload for trainset section extraction.

    The analyzer returns trainset rows for the requested scope as a normalized
    DataFrame suitable for table/plot rendering.

    Fields
    -----
    request : GetTrainsetDataRequest
        Request object used to generate this result.
    table : pandas.DataFrame
        Extracted trainset rows. For ``section="all"``, includes a leading
        ``section`` column identifying each source section.

    Notes
    -----
    Column schemas vary by section because each trainset block has distinct
    row fields.

    Examples
    -----
    ```python
    row = {
        "section": "energy",
        "line_number": 142,
        "op1": "+",
        "id1": "bulk_1",
        "n1": 1.0,
        "lit": -15.4,
    }
    ```
    The sample row illustrates one ENERGY entry in an all-sections output.
    """

    table: pd.DataFrame
    request: TrainsetDataRequest


@register_task("trainset_data", label="Trainset Data")
class TrainsetDataTask(AnalysisTask):
    """Return trainset rows for one section or all sections."""

    required_data = ForceFieldOptimizationTrainingSetData

    @staticmethod
    def recommended_presentations(
        _result: TrainsetDataResult,
        payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        """Recommend table and fallback plot views for trainset row outputs.

        Always emits a table view and adds a simple plot based on detected
        numeric columns and standard trainset axis fields.

        Works on
        Analyzer task output for ``trainset_data``.

        Parameters
        -----
        _result : TrainsetDataResult
            Typed analyzer result instance (unused by current logic).
        payload : dict[str, Any]
            Serialized payload expected to contain a ``table`` list.

        Returns
        -----
        list[PresentationSpec]
            Recommended presentation specs for trainset tables.

        Examples
        -----
        ```python
        specs = GetTrainsetDataTask.recommended_presentations(
            _result,
            {"table": [{"section": "energy", "line_number": 142, "lit": -15.4}]},
        )
        ```
        The returned specs include a table and a default numeric plot.
        """
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
        request: TrainsetDataRequest,
        reporter=None,
    ) -> TrainsetDataResult:
        """Run trainset section extraction for the requested scope.

        Resolves the section selector, materializes either one section table or
        a concatenated all-section table, and returns a typed analyzer result.

        Works on
        ``ForceFieldOptimizationTrainingSetData``.

        Parameters
        -----
        data : ForceFieldOptimizationTrainingSetData
            Parsed trainset data bundle.
        request : TrainsetDataRequest
            Request with section selector.
        reporter : Any, optional
            Progress callback accepted by analyzer tasks; unused here.

        Returns
        -----
        TrainsetDataResult
            Result containing the extracted trainset table.

        Examples
        -----
        ```python
        result = GetTrainsetDataTask().run(data, GetTrainsetDataRequest(section="all"))
        ```
        The returned table contains all supported sections with a ``section`` label.
        """
        table = _build_trainset_data_table(data, section=request.section)
        return TrainsetDataResult(table=table, request=request)


@dataclass
class TrainsetGroupCommentsRequest(BaseRequest):
    """Request payload for unique trainset group-comment extraction.

    This request selects one trainset section or all sections when collecting
    unique non-empty ``group_comment`` annotations.

    Fields
    -----
    section : str
        Section selector. Use ``"all"`` for every section, or one of
        ``"charge"``, ``"heatfo"``, ``"geometry"``, ``"cell_parameters"``,
        ``"energy"`` for section-scoped comment extraction.

    Examples
    -----
    ```python
    request = TrainsetGroupCommentsRequest(section="geometry")
    ```
    The request limits comment extraction to GEOMETRY rows.
    """

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
    """Result payload containing grouped trainset comments.

    The analyzer returns unique per-section comments with earliest line numbers
    when available, suitable for quality checks and section-level summaries.

    Fields
    -----
    request : TrainsetGroupCommentsRequest
        Request object used to generate this result.
    table : pandas.DataFrame
        Table with columns ``section``, ``group_comment``, ``line_number``,
        and ``count`` (currently set to ``1`` per unique comment row).

    Examples
    -----
    ```python
    row = {
        "section": "energy",
        "group_comment": "equation_of_state_reference_set",
        "line_number": 142,
        "count": 1,
    }
    ```
    The sample indicates one unique ENERGY comment entry.
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
        """Recommend table and section-count plots for grouped comments output.

        Emits a table view for all outputs and adds a section-count plot when
        section labels are available in serialized rows.

        Works on
        Analyzer task output for ``trainset_group_comments``.

        Parameters
        -----
        _result : TrainsetGroupCommentsResult
            Typed analyzer result instance (unused in current selection logic).
        payload : dict[str, Any]
            Serialized payload expected to include ``table`` rows.

        Returns
        -----
        list[PresentationSpec]
            Presentation specs suitable for comments tables and count plots.

        Examples
        -----
        ```python
        specs = TrainsetGroupCommentsTask.recommended_presentations(
            _result,
            {"table": [{"section": "energy", "group_comment": "eos", "count": 1}]},
        )
        ```
        The output includes a table and a comment-count-by-section plot.
        """
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
        """Run unique group-comment extraction from parsed trainset data.

        Builds the grouped-comment table across sections, then optionally
        filters rows to the request-selected section.

        Works on
        ``ForceFieldOptimizationTrainingSetData``.

        Parameters
        -----
        data : ForceFieldOptimizationTrainingSetData
            Parsed trainset data source.
        request : TrainsetGroupCommentsRequest
            Request containing section scope for comment extraction.
        reporter : Any, optional
            Progress callback accepted by analyzer tasks; unused here.

        Returns
        -----
        TrainsetGroupCommentsResult
            Result containing grouped comments and metadata.

        Examples
        -----
        ```python
        result = TrainsetGroupCommentsTask().run(
            data,
            TrainsetGroupCommentsRequest(section="all"),
        )
        ```
        The result table contains unique comments from all supported sections.
        """
        table = _get_trainset_group_comments(data)
        section_key = _normalize_trainset_section(request.section)
        if section_key != "all" and not table.empty and "section" in table.columns:
            table = table.loc[table["section"] == section_key.lower()].copy().reset_index(drop=True)
        return TrainsetGroupCommentsResult(table=table, request=request)


__all__ = [
    "TrainsetDataRequest",
    "TrainsetDataResult",
    "TrainsetDataTask",
    "TrainsetGroupCommentsRequest",
    "TrainsetGroupCommentsResult",
    "TrainsetGroupCommentsTask",
]
