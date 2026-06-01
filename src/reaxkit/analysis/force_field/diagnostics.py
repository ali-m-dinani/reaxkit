"""Analyze parameter-optimization diagnostics in an engine-agnostic format.

This module extracts and normalizes optimization-diagnostic records, including
parameter-change signals and section-aware summaries used during force-field
tuning. It is scoped to diagnostic artifacts and does not mutate force-field
parameters.

When using Standalone ReaxFF for force field optimization, this data will be available in 'fort.79' output file.

**Usage context**

- Optimization debugging: Inspect parameter update behavior across iterations.
- Section diagnostics: Break down signals by force-field parameter sections.
- QA reporting: Export normalized diagnostic tables for review dashboards.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
import re
from typing import Any

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis.force_field.force_field import FFieldDataRequest, FFieldDataTask
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import (
    ForceFieldOptimizationDiagnosticBundleData,
    ForceFieldOptimizationDiagnosticData,
    ForceFieldParametersData,
)
from reaxkit.presentation.specs import PresentationSpec

_SECTION_NUM_MAP: dict[int, tuple[str, str]] = {
    1: ("general", "general"),
    2: ("atom", "atom"),
    3: ("bond", "bond"),
    4: ("off_diagonal", "off_diagonal"),
    5: ("angle", "angle"),
    6: ("torsion", "torsion"),
    7: ("hbond", "hbond"),
}

_SECTION_INDEX_COLS: dict[str, list[str]] = {
    "general": [],
    "atom": ["symbol"],
    "bond": ["i", "j"],
    "off_diagonal": ["i", "j"],
    "angle": ["i", "j", "k"],
    "torsion": ["i", "j", "k", "l"],
    "hbond": ["i", "j", "k"],
}


def _diagnostic_frame(data: ForceFieldOptimizationDiagnosticData) -> pd.DataFrame:
    """Build the base diagnostics table from parsed optimization diagnostics."""
    return pd.DataFrame(
        {
            "identifier": pd.Series(data.identifiers, dtype=object),
            "value1": pd.Series(data.value1, dtype=float),
            "value2": pd.Series(data.value2, dtype=float),
            "value3": pd.Series(data.value3, dtype=float),
            "diff1": pd.Series(data.diff1, dtype=float),
            "diff2": pd.Series(data.diff2, dtype=float),
            "diff3": pd.Series(data.diff3, dtype=float),
            "a": pd.Series(data.a, dtype=float),
            "b": pd.Series(data.b, dtype=float),
            "c": pd.Series(data.c, dtype=float),
            "parabol_min": pd.Series(data.parabol_min, dtype=float),
            "parabol_min_diff": pd.Series(data.parabol_min_diff, dtype=float),
            "value4": pd.Series(data.value4, dtype=float),
            "diff4": pd.Series(data.diff4, dtype=float),
        }
    )


def _diagnostic_sensitivity_table(
    data: ForceFieldOptimizationDiagnosticData,
) -> pd.DataFrame:
    """Return raw diagnostic data augmented with relative sensitivities."""
    df = _diagnostic_frame(data)
    diff3 = pd.to_numeric(df["diff3"], errors="coerce").replace(0.0, np.nan)

    out = df.copy()
    out["sensitivity1/3"] = pd.to_numeric(out["diff1"], errors="coerce") / diff3
    out["sensitivity2/3"] = pd.to_numeric(out["diff2"], errors="coerce") / diff3
    out["sensitivity4/3"] = pd.to_numeric(out["diff4"], errors="coerce") / diff3
    out["min_sensitivity"] = out[["sensitivity1/3", "sensitivity2/3", "sensitivity4/3"]].min(axis=1)
    out["max_sensitivity"] = out[["sensitivity1/3", "sensitivity2/3", "sensitivity4/3"]].max(axis=1)
    return out


def _param_columns_for_section(sec_df: pd.DataFrame, section_key: str) -> list[str]:
    """Return parameter columns that are not section index/identity fields."""
    idx_cols = set(_SECTION_INDEX_COLS.get(section_key, []))
    return [c for c in sec_df.columns if c not in idx_cols]


def _parse_identifier_triplet(identifier: Any) -> tuple[int, int, int] | None:
    """Parse ``(section, line, parameter)`` integers from an identifier string."""
    text = str(identifier).strip()
    nums = [int(x) for x in re.findall(r"-?\d+", text)]
    if len(nums) < 3:
        return None
    return nums[0], nums[1], nums[2]


def _interpret_identifier_details(
    identifier: Any,
    *,
    force_field: ForceFieldParametersData,
    cache: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Resolve identifier triplets into force-field section/parameter metadata."""
    out: dict[str, Any] = {
        "ff_section": pd.NA,
        "ff_section_line": pd.NA,
        "ff_parameter": pd.NA,
        "component": pd.NA,
        "ffield_section_name": pd.NA,
        "ffield_value": pd.NA,
        "term": pd.NA,
    }
    parsed = _parse_identifier_triplet(identifier)
    if parsed is None:
        return out
    sec_num, line_1b, par_1b = parsed
    out["ff_section"] = int(sec_num)
    out["ff_section_line"] = int(line_1b)
    out["ff_parameter"] = int(par_1b)
    if sec_num not in _SECTION_NUM_MAP:
        return out
    section_key, section_name = _SECTION_NUM_MAP[sec_num]
    out["ffield_section_name"] = section_name

    if section_key not in cache:
        cache[section_key] = FFieldDataTask().run(
            force_field,
            FFieldDataRequest(
                section=section_name,
                interpret=section_key not in {"general", "atom"},
            ),
        ).table

    sec_df = cache[section_key]
    row_idx = int(line_1b) - 1
    if row_idx < 0 or row_idx >= len(sec_df):
        return out

    param_cols = _param_columns_for_section(sec_df, section_key)
    param_cols = [c for c in param_cols if not (str(c).endswith("_symbol") or c == "term")]
    par_idx = int(par_1b) - 1
    if par_idx < 0 or par_idx >= len(param_cols):
        return out

    param_name = str(param_cols[par_idx])
    sec_row = sec_df.iloc[row_idx]
    out["component"] = param_name
    out["ffield_value"] = sec_row.get(param_name, pd.NA)
    out["term"] = sec_row.get("term", pd.NA)
    return out


def _with_interpreted_identifiers(
    table: pd.DataFrame,
    *,
    force_field: ForceFieldParametersData,
) -> pd.DataFrame:
    """Insert interpreted identifier metadata columns into a diagnostics table."""
    out = table.copy()
    sec_cache: dict[str, pd.DataFrame] = {}
    details = out["identifier"].map(
        lambda raw: _interpret_identifier_details(raw, force_field=force_field, cache=sec_cache)
    )
    details_df = pd.DataFrame(details.tolist())
    id_loc = out.columns.get_loc("identifier")
    for offset, col in enumerate(
        [
            "ff_section",
            "ff_section_line",
            "ff_parameter",
            "component",
            "ffield_section_name",
            "ffield_value",
            "term",
        ]
    ):
        out.insert(id_loc + 1 + offset, col, details_df[col])
    out = out.drop(columns=["identifier"])
    return out


@dataclass
class FFieldOptimizationDiagnosticRequest(BaseRequest):
    """Request payload for optimization diagnostics analysis.

    This request controls whether parsed diagnostic identifiers are returned as
    raw values or interpreted into section/parameter metadata using force-field
    parameter tables from the same analysis bundle.

    Fields
    -----
    interpret : bool
        If ``True``, decode identifier triplets into descriptive columns such as
        section name, parameter component, and interpreted term labels.

    Examples
    -----
    ```python
    request = ParameterOptimizationDiagnosticRequest(interpret=True)
    ```
    The request asks for interpreted identifier metadata in the output table.
    """

    interpret: bool = dc_field(
        default=False,
        metadata={
            "label": "Interpret",
            "help": (
                "If true, interpret identifier triplets (section, line, parameter) "
                "into descriptive force-field parameter names using loaded force-field data."
            ),
            "choices": [True, False],
        },
    )


@dataclass
class FFieldOptimizationDiagnosticResult(BaseResult):
    """Result payload for parameter-optimization diagnostics.

    The analyzer returns raw diagnostic values plus derived sensitivity ratios,
    with optional interpreted force-field identifier metadata when requested.

    Fields
    -----
    request : ParameterOptimizationDiagnosticRequest
        Request object used to generate this result.
    table : pandas.DataFrame
        Table containing raw diagnostic columns, derived sensitivity columns,
        and optional interpreted identifier metadata columns.

    Notes
    -----
    Sensitivity ratios are computed as ``diff1/diff3``, ``diff2/diff3``, and
    ``diff4/diff3`` after coercion to numeric values; zero ``diff3`` values are
    treated as missing to avoid division-by-zero artifacts.

    Examples
    -----
    ```python
    row = {
        "identifier": "3 12 4",
        "sensitivity1/3": 0.8,
        "sensitivity2/3": 1.2,
        "sensitivity4/3": 0.5,
        "min_sensitivity": 0.5,
        "max_sensitivity": 1.2,
    }
    ```
    ``min_sensitivity`` and ``max_sensitivity`` summarize each row's ratio span.
    """

    table: pd.DataFrame
    request: FFieldOptimizationDiagnosticRequest


@register_task("parameter_optimization_diagnostic", label="Parameter Optimization Diagnostic")
class FFieldOptimizationDiagnosticTask(AnalysisTask):
    """Return sensitivity diagnostics derived from parameter-update diagnostics."""

    required_data = ForceFieldOptimizationDiagnosticBundleData

    @staticmethod
    def recommended_presentations(
        _result: FFieldOptimizationDiagnosticResult, payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        """Recommend table and sensitivity plot views for diagnostics output.

        Produces a table view for all outputs and adds a default
        ``min_sensitivity`` vs ``identifier`` plot when required columns exist.

        Works on
        Analyzer task output for ``parameter_optimization_diagnostic``.

        Parameters
        -----
        _result : FFieldOptimizationDiagnosticResult
            Typed analyzer result instance (unused for current selection logic).
        payload : dict[str, Any]
            Serialized analyzer payload expected to include ``table`` rows.

        Returns
        -----
        list[PresentationSpec]
            Recommended renderer specifications for diagnostics outputs.

        Examples
        -----
        ```python
        specs = ParameterOptimizationDiagnosticTask.recommended_presentations(
            _result,
            {"table": [{"identifier": "3 12 4", "min_sensitivity": 0.5}]},
        )
        ```
        The returned list includes a table and a one-series sensitivity plot.
        """
        rows = payload.get("table")
        if not isinstance(rows, list) or not rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = rows[0] if isinstance(rows[0], dict) else {}
        if "identifier" not in sample or "min_sensitivity" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="Min Sensitivity vs Identifier",
                mapping={"x_col": "identifier", "y_col": "min_sensitivity", "group_by_col": ""},
                options={
                    "title": "Min Sensitivity vs Identifier",
                    "xlabel": "identifier",
                    "ylabel": "min_sensitivity",
                    "legend": False,
                },
                view_type="plot2d",
            ),
        ]

    def run(
        self,
        data: ForceFieldOptimizationDiagnosticBundleData,
        request: FFieldOptimizationDiagnosticRequest,
        reporter=None,
    ) -> FFieldOptimizationDiagnosticResult:
        """Run diagnostics analysis and optional identifier interpretation.

        Builds the sensitivity-augmented diagnostics table from parsed
        optimization diagnostics and, when requested, enriches identifiers using
        parsed force-field parameter data from the same bundle.

        Works on
        ``ForceFieldOptimizationDiagnosticBundleData``.

        Parameters
        -----
        data : ForceFieldOptimizationDiagnosticBundleData
            Bundle containing diagnostics and force-field parameter records.
        request : FFieldOptimizationDiagnosticRequest
            Request controlling identifier interpretation.
        reporter : Any, optional
            Progress callback accepted by the analyzer interface; unused here.

        Returns
        -----
        FFieldOptimizationDiagnosticResult
            Result containing raw/derived diagnostics and optional metadata.

        Examples
        -----
        ```python
        result = ParameterOptimizationDiagnosticTask().run(
            data,
            ParameterOptimizationDiagnosticRequest(interpret=False),
        )
        ```
        The output contains sensitivity columns for each diagnostics row.
        """
        diagnostic_data = data.diagnostics
        if diagnostic_data is None:
            raise ValueError("ForceFieldOptimizationDiagnosticBundleData.diagnostics is required.")
        table = _diagnostic_sensitivity_table(diagnostic_data)
        if bool(request.interpret):
            table = _with_interpreted_identifiers(table, force_field=data.force_field_parameters)
        return FFieldOptimizationDiagnosticResult(table=table, request=request)


__all__ = [
    "FFieldOptimizationDiagnosticRequest",
    "FFieldOptimizationDiagnosticResult",
    "FFieldOptimizationDiagnosticTask",
]
