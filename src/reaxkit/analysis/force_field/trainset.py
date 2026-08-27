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
import re
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
    """Collect every comment-block occurrence, including empty blocks."""
    tables = _get_trainset_section_tables(data)
    rows: list[dict[str, object]] = []
    for section_name, df in tables.items():
        if df.empty or "group_comment" not in df.columns:
            continue
        work = df.copy()
        work["group_comment"] = work["group_comment"].fillna("").astype(str).str.strip()
        work["line_number"] = pd.to_numeric(
            work.get("line_number", pd.Series(index=work.index, dtype=float)),
            errors="coerce",
        )
        work["group_comment_line_number"] = pd.to_numeric(
            work.get(
                "group_comment_line_number",
                pd.Series(index=work.index, dtype=float),
            ),
            errors="coerce",
        )
        work = work.sort_values("line_number", kind="stable", na_position="last")

        blocks: list[list[object]] = []
        current_indices: list[object] = []
        previous_marker: object = None
        previous_comment: str | None = None
        previous_line: float | None = None
        for index, entry in work.iterrows():
            marker_value = entry.get("group_comment_line_number")
            marker = None if pd.isna(marker_value) else float(marker_value)
            comment = str(entry.get("group_comment", ""))
            line_value = entry.get("line_number")
            line_number = None if pd.isna(line_value) else float(line_value)
            new_block = not current_indices
            if current_indices and (marker != previous_marker or comment != previous_comment):
                new_block = True
            elif (
                current_indices
                and marker is None
                and previous_marker is None
                and line_number is not None
                and previous_line is not None
                and line_number > previous_line + 1
            ):
                new_block = True
            if new_block and current_indices:
                blocks.append(current_indices)
                current_indices = []
            current_indices.append(index)
            previous_marker = marker
            previous_comment = comment
            previous_line = line_number
        if current_indices:
            blocks.append(current_indices)

        for occurrence, indices in enumerate(blocks, start=1):
            block = work.loc[indices]
            comment = str(block.iloc[0].get("group_comment", ""))
            marker = pd.to_numeric(
                block["group_comment_line_number"], errors="coerce"
            ).dropna()
            entry_lines = pd.to_numeric(block["line_number"], errors="coerce").dropna()
            comment_line = marker.iloc[0] if not marker.empty else pd.NA
            first_entry_line = entry_lines.min() if not entry_lines.empty else pd.NA
            last_entry_line = entry_lines.max() if not entry_lines.empty else pd.NA
            rows.append(
                {
                    "section": section_name.lower(),
                    "occurrence": occurrence,
                    "group_comment": comment,
                    "type of data": _classify_trainset_comment_block(
                        section_name, block, comment
                    ),
                    "identifiers": _trainset_block_identifiers(block),
                    "line_number": (
                        comment_line if pd.notna(comment_line) else first_entry_line
                    ),
                    "comment_line_number": comment_line,
                    "first_entry_line_number": first_entry_line,
                    "last_entry_line_number": last_entry_line,
                    "count": len(block),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for column in (
        "line_number",
        "comment_line_number",
        "first_entry_line_number",
        "last_entry_line_number",
    ):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
    return out.sort_values(["line_number"], kind="stable", na_position="last").reset_index(drop=True)


def _trainset_block_identifiers(block: pd.DataFrame) -> str:
    """Return stable unique identifiers referenced by one comment block."""
    columns = ["iden"] if "iden" in block.columns else []
    columns.extend(
        sorted(
            (column for column in block.columns if re.fullmatch(r"id\d+", str(column))),
            key=lambda column: int(str(column)[2:]),
        )
    )
    values: list[str] = []
    for column in columns:
        for value in block[column]:
            if pd.isna(value):
                continue
            identifier = str(value).strip()
            if identifier and identifier not in values:
                values.append(identifier)
    return "; ".join(values)


def _identifier_suggests_eos(identifier: object) -> bool:
    text = str(identifier).strip().lower()
    return bool(
        re.search(r"(?:^|[_\-.])(bulk|volume|eos)(?:[_\-.]|$)", text)
        or re.search(r"(?:^|[_\-.])c\d+(?:[_\-.]|$)", text)
    )


def _geometry_block_type(block: pd.DataFrame) -> str:
    types: list[str] = []
    for _, entry in block.iterrows():
        atom_count = sum(
            pd.notna(entry.get(column)) for column in ("at1", "at2", "at3", "at4")
        )
        kind = {
            0: "rmsg",
            2: "bond",
            3: "valence_angle",
            4: "torsion_angle",
        }.get(atom_count, "geometry")
        if kind not in types:
            types.append(kind)
    return ", ".join(types) if types else "geometry"


def _classify_trainset_comment_block(
    section_name: str,
    block: pd.DataFrame,
    comment: str,
) -> str:
    """Infer the training-data kind from its section, comment, and entries."""
    if section_name == "CHARGE":
        return "charge"
    if section_name == "HEATFO":
        return "heatfo"
    if section_name == "GEOMETRY":
        return _geometry_block_type(block)
    if section_name == "CELL_PARAMETERS":
        return "cell_parameters"
    if section_name != "ENERGY":
        return section_name.lower()

    normalized_comment = str(comment).lower()
    if re.search(r"\b(?:heat\s+of\s+formation|heatfo)\b", normalized_comment):
        return "heatfo"
    if re.search(r"\brestraint\b", normalized_comment):
        return "restraint"

    identifier_columns = sorted(
        (column for column in block.columns if re.fullmatch(r"id\d+", str(column))),
        key=lambda column: int(str(column)[2:]),
    )
    identifiers = [
        value
        for column in identifier_columns
        for value in block[column]
        if pd.notna(value) and str(value).strip()
    ]
    if re.search(r"\b(?:eos|volume|bulk)\b", normalized_comment) or any(
        _identifier_suggests_eos(identifier) for identifier in identifiers
    ):
        return "eos"

    operand_counts = [
        sum(pd.notna(entry.get(column)) and bool(str(entry.get(column)).strip()) for column in identifier_columns)
        for _, entry in block.iterrows()
    ]
    if operand_counts and all(count >= 3 for count in operand_counts):
        return "reaction_energy"
    if operand_counts and all(count == 2 for count in operand_counts):
        identifier_counts = pd.Series([str(value).strip() for value in identifiers]).value_counts()
        return "energy_curve" if (identifier_counts > 1).any() else "energy_difference"
    return "energy"


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


def _select_trainset_section_tables(
    data: ForceFieldOptimizationTrainingSetData,
    section: str,
) -> dict[str, pd.DataFrame]:
    """Return native-schema tables for the requested trainset section scope."""
    tables = _get_trainset_section_tables(data)
    section_key = _normalize_trainset_section(section)
    if section_key == "all":
        return {name: table.reset_index(drop=True) for name, table in tables.items()}
    return {section_key: tables[section_key].reset_index(drop=True)}


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
    section_tables : dict[str, pandas.DataFrame]
        Requested trainset sections as separate, native-schema tables keyed by
        canonical section name. These tables support section-by-section export
        without the empty columns introduced by concatenating unlike schemas.

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
    section_tables: dict[str, pd.DataFrame] = dc_field(default_factory=dict)


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
        section_tables = _select_trainset_section_tables(data, section=request.section)
        return TrainsetDataResult(
            table=table,
            request=request,
            section_tables=section_tables,
        )


@dataclass
class TrainsetGroupCommentsRequest(BaseRequest):
    """Request payload for occurrence-level trainset group-comment extraction.

    This request selects one trainset section or all sections when collecting
    every ``group_comment`` block, including repeated and empty annotations.

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

    The analyzer returns every per-section comment occurrence with its comment
    and entry line ranges, inferred data type, identifiers, and entry count.

    Fields
    -----
    request : TrainsetGroupCommentsRequest
        Request object used to generate this result.
    table : pandas.DataFrame
        Occurrence table including ``section``, ``group_comment``,
        ``type of data``, source lines, identifiers, and entry ``count``.

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
    """Return every trainset group-comment occurrence by section."""

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
        """Run occurrence-level group-comment extraction from parsed data.

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
