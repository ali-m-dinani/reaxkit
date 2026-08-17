"""Prepare bounded parameter samples for optimization-diagnostic beeswarm plots."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field as dc_field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis.force_field.diagnostics import (
    _SECTION_NUM_MAP,
    _interpret_identifier_details,
    _parse_identifier_triplet,
)
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ForceFieldOptimizationDiagnosticPlotData


_SAMPLES = (
    ("Sample 1", "value1", "diff1"),
    ("Sample 2", "value2", "diff2"),
    ("Sample 3", "value3", "diff3"),
    ("Parabola minimum", "parabol_min", "parabol_min_diff"),
    ("Final value", "value4", "diff4"),
)

_SAMPLE_COLUMNS = [
    "parameter_key",
    "parameter_label",
    "sample_name",
    "parameter_value",
    "normalized_value",
    "objective_value",
    "diagnostic_row",
    "plot_row",
]

_PARAMETER_COLUMNS = [
    "parameter_key",
    "ff_section",
    "ff_section_line",
    "ff_parameter",
    "parameter_label",
    "section_name",
    "term",
    "component",
    "lower_bound",
    "upper_bound",
    "search_interval",
    "starting_value",
    "final_value",
    "objective_min",
    "objective_max",
    "color_min",
    "color_max",
    "sample_count",
    "plot_row",
]


def _number(value: Any) -> float | None:
    """Return a finite float for a scalar value, otherwise ``None``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _color_bounds(values: Iterable[float]) -> tuple[float, float]:
    """Return non-degenerate color limits for one or more objective values."""
    finite = [float(value) for value in values if np.isfinite(float(value))]
    lower, upper = min(finite), max(finite)
    if lower == upper:
        padding = max(abs(lower) * 1e-6, 1e-9)
        return lower - padding, upper + padding
    return lower, upper


def _text(value: Any) -> str:
    """Return display text without exposing pandas missing-value sentinels."""
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _parameter_label(details: dict[str, Any], pointer: tuple[int, int, int]) -> str:
    """Build a human-readable force-field parameter label."""
    section_name = _text(details.get("ffield_section_name"))
    if not section_name:
        section_name = _SECTION_NUM_MAP.get(pointer[0], (str(pointer[0]), str(pointer[0])))[1]
    section_title = section_name.replace("_", " ").title()
    term = _text(details.get("term"))
    component = _text(details.get("component"))
    prefix = " ".join(part for part in (section_title, term) if part)
    return f"{prefix} - {component}" if component else (prefix or " ".join(map(str, pointer)))


def _parameter_definitions(data: ForceFieldOptimizationDiagnosticPlotData) -> dict[tuple[int, int, int], dict[str, Any]]:
    """Index optimization bounds and increments by numeric force-field pointer."""
    definitions: dict[tuple[int, int, int], dict[str, Any]] = {}
    params = data.optimization_parameters
    for values in zip(
        params.ff_section,
        params.ff_section_line,
        params.ff_parameter,
        params.search_interval,
        params.min_value,
        params.max_value,
        strict=True,
    ):
        try:
            key = (int(values[0]), int(values[1]), int(values[2]))
        except (TypeError, ValueError):
            continue
        definitions[key] = {
            "search_interval": _number(values[3]),
            "lower_bound": _number(values[4]),
            "upper_bound": _number(values[5]),
        }
    return definitions


def _diagnostic_records(data: ForceFieldOptimizationDiagnosticPlotData) -> list[dict[str, Any]]:
    """Return row-aligned diagnostic values while preserving source order."""
    diagnostics = data.diagnostics
    records: list[dict[str, Any]] = []
    fields = [name for _, name, _ in _SAMPLES] + [name for _, _, name in _SAMPLES]
    for index, identifier in enumerate(diagnostics.identifiers):
        record = {"identifier": identifier, "diagnostic_row": index + 1}
        for field in fields:
            record[field] = getattr(diagnostics, field)[index]
        records.append(record)
    return records


def build_diagnostic_beeswarm_tables(
    data: ForceFieldOptimizationDiagnosticPlotData,
    *,
    sort_by: str = "parameter",
    global_objective_scale: bool = False,
    top: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Build normalized diagnostic samples and their parameter summaries.

    Parameter values are normalized with the declared optimization bounds, not
    the minimum and maximum values observed in the diagnostic output.
    """
    if sort_by not in {"parameter", "final", "starting"}:
        raise ValueError("sort_by must be one of: parameter, final, starting")
    if int(top) < 0:
        raise ValueError("top must be zero or a positive integer")

    definitions = _parameter_definitions(data)
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    excluded: set[str] = set()
    for record in _diagnostic_records(data):
        pointer = _parse_identifier_triplet(record["identifier"])
        if pointer is None:
            excluded.add(str(record["identifier"]))
            continue
        grouped[pointer].append(record)

    sample_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    interpretation_cache: dict[str, pd.DataFrame] = {}
    for pointer, records in grouped.items():
        definition = definitions.get(pointer)
        if definition is None:
            excluded.add(" ".join(map(str, pointer)))
            continue
        lower = definition["lower_bound"]
        upper = definition["upper_bound"]
        if lower is None or upper is None or lower == upper:
            excluded.add(" ".join(map(str, pointer)))
            continue

        details = _interpret_identifier_details(
            records[0]["identifier"],
            force_field=data.force_field_parameters,
            cache=interpretation_cache,
        )
        parameter_key = " ".join(map(str, pointer))
        label = _parameter_label(details, pointer)
        group_samples: list[dict[str, Any]] = []
        for record in records:
            for sample_name, value_field, objective_field in _SAMPLES:
                value = _number(record[value_field])
                objective = _number(record[objective_field])
                if value is None or objective is None:
                    continue
                normalized = min(1.0, max(0.0, (value - lower) / (upper - lower)))
                group_samples.append({
                    "parameter_key": parameter_key,
                    "parameter_label": label,
                    "sample_name": sample_name,
                    "parameter_value": value,
                    "normalized_value": normalized,
                    "objective_value": objective,
                    "diagnostic_row": int(record["diagnostic_row"]),
                })
        if not group_samples:
            excluded.add(parameter_key)
            continue

        objectives = [row["objective_value"] for row in group_samples]
        objective_min, objective_max = _color_bounds(objectives)
        final_value = next(
            (
                value
                for value in (_number(record["value4"]) for record in reversed(records))
                if value is not None
            ),
            None,
        )
        parameter_rows.append({
            "parameter_key": parameter_key,
            "pointer": pointer,
            "ff_section": pointer[0],
            "ff_section_line": pointer[1],
            "ff_parameter": pointer[2],
            "parameter_label": label,
            "section_name": _text(details.get("ffield_section_name")),
            "term": _text(details.get("term")),
            "component": _text(details.get("component")),
            "lower_bound": lower,
            "upper_bound": upper,
            "search_interval": definition["search_interval"],
            "starting_value": _number(details.get("ffield_value")),
            "final_value": final_value,
            "objective_min": objective_min,
            "objective_max": objective_max,
            "objective_span": objective_max - objective_min,
            "sample_count": len(group_samples),
        })
        sample_rows.extend(group_samples)

    if top:
        selected = sorted(
            parameter_rows,
            key=lambda row: (-float(row["objective_span"]), row["pointer"]),
        )[: int(top)]
        selected_keys = {row["parameter_key"] for row in selected}
        parameter_rows = selected
        sample_rows = [row for row in sample_rows if row["parameter_key"] in selected_keys]

    def numeric_key(row: dict[str, Any], field: str) -> tuple[bool, float, tuple[int, int, int]]:
        value = row[field]
        return value is None, float(value or 0.0), row["pointer"]

    sort_keys = {
        "parameter": lambda row: (row["pointer"], row["parameter_label"]),
        "final": lambda row: numeric_key(row, "final_value"),
        "starting": lambda row: numeric_key(row, "starting_value"),
    }
    parameter_rows.sort(key=sort_keys[sort_by])

    if parameter_rows:
        if global_objective_scale:
            global_min, global_max = _color_bounds(row["objective_value"] for row in sample_rows)
        else:
            global_min = global_max = None
        plot_rows = {row["parameter_key"]: index for index, row in enumerate(parameter_rows)}
        for index, row in enumerate(parameter_rows):
            row["plot_row"] = index
            row["color_min"] = global_min if global_objective_scale else row["objective_min"]
            row["color_max"] = global_max if global_objective_scale else row["objective_max"]
            row.pop("pointer", None)
            row.pop("objective_span", None)
        for row in sample_rows:
            row["plot_row"] = plot_rows[row["parameter_key"]]

    samples = pd.DataFrame(sample_rows, columns=_SAMPLE_COLUMNS)
    parameters = pd.DataFrame(parameter_rows, columns=_PARAMETER_COLUMNS)
    return samples, parameters, len(excluded)


@dataclass
class FFieldOptimizationDiagnosticBeeswarmRequest(BaseRequest):
    """Configure bounded diagnostic sample preparation."""

    sort_by: str = dc_field(
        default="parameter",
        metadata={
            "label": "Sort rows by",
            "help": "Sort parameter rows by numeric pointer, final value, or starting value.",
            "choices": ["parameter", "final", "starting"],
        },
    )
    global_objective_scale: bool = dc_field(
        default=False,
        metadata={
            "label": "Global objective scale",
            "help": "Use one objective-function color range for every parameter row.",
            "choices": [True, False],
        },
    )
    top: int = dc_field(
        default=0,
        metadata={
            "label": "Top parameters",
            "help": "Keep the parameters with the widest objective ranges; zero keeps all.",
        },
    )


@dataclass
class FFieldOptimizationDiagnosticBeeswarmResult(BaseResult):
    """Normalized diagnostic sample rows and parameter-level plot metadata."""

    table: pd.DataFrame
    parameters: pd.DataFrame
    excluded_parameters: int
    request: FFieldOptimizationDiagnosticBeeswarmRequest


@register_task(
    "parameter_optimization_diagnostic_beeswarm",
    label="Parameter Optimization Diagnostic Beeswarm",
)
class FFieldOptimizationDiagnosticBeeswarmTask(AnalysisTask):
    """Prepare the bounded samples used by the diagnostic beeswarm renderer."""

    required_data = ForceFieldOptimizationDiagnosticPlotData

    def run(
        self,
        data: ForceFieldOptimizationDiagnosticPlotData,
        request: FFieldOptimizationDiagnosticBeeswarmRequest,
        reporter=None,
    ) -> FFieldOptimizationDiagnosticBeeswarmResult:
        """Build plot-ready sample and parameter tables from typed inputs."""
        _ = reporter
        samples, parameters, excluded = build_diagnostic_beeswarm_tables(
            data,
            sort_by=str(request.sort_by),
            global_objective_scale=bool(request.global_objective_scale),
            top=int(request.top),
        )
        return FFieldOptimizationDiagnosticBeeswarmResult(
            table=samples,
            parameters=parameters,
            excluded_parameters=excluded,
            request=request,
        )


__all__ = [
    "FFieldOptimizationDiagnosticBeeswarmRequest",
    "FFieldOptimizationDiagnosticBeeswarmResult",
    "FFieldOptimizationDiagnosticBeeswarmTask",
    "build_diagnostic_beeswarm_tables",
]
