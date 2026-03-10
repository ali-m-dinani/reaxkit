"""Geometry-optimization data extraction tasks."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Sequence

import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.alias import normalize_choice, resolve_alias_from_columns
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import GeometryOptimizationProgressData
from reaxkit.presentation.specs import PresentationSpec

_GEOMETRY_OPT_CANONICAL = ("iter", "E_pot", "T", "T_set", "RMSG", "nfc")
_GEOMETRY_OPT_COMPONENT_CHOICES = ("E_pot", "T", "T_set", "RMSG", "nfc")


def _geometry_optimization_frame(data: GeometryOptimizationProgressData) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "iter": pd.Series(data.optimization_iterations, dtype=int),
            "E_pot": (
                pd.Series(data.potential_energy, dtype=float)
                if data.potential_energy is not None
                else pd.Series([pd.NA] * len(data.optimization_iterations))
            ),
            "T": (
                pd.Series(data.temperature, dtype=float)
                if data.temperature is not None
                else pd.Series([pd.NA] * len(data.optimization_iterations))
            ),
            "T_set": (
                pd.Series(data.temperature_setpoint, dtype=float)
                if data.temperature_setpoint is not None
                else pd.Series([pd.NA] * len(data.optimization_iterations))
            ),
            "RMSG": (
                pd.Series(data.rms_gradient, dtype=float)
                if data.rms_gradient is not None
                else pd.Series([pd.NA] * len(data.optimization_iterations))
            ),
            "nfc": (
                pd.Series(data.n_force_calls, dtype="Int64")
                if data.n_force_calls is not None
                else pd.Series([pd.NA] * len(data.optimization_iterations), dtype="Int64")
            ),
        }
    )


def _build_geometry_optimization_table(
    *,
    data: GeometryOptimizationProgressData,
    components: Sequence[str] | None = None,
    include_geo_descriptor: bool = False,
) -> pd.DataFrame:
    """Build selected geometry-optimization components as a long table."""
    df = _geometry_optimization_frame(data)
    available = list(df.columns)

    if components is None or len(components) == 0:
        wanted_canon = list(_GEOMETRY_OPT_COMPONENT_CHOICES)
    else:
        wanted_canon = [normalize_choice(c, domain="fort57.md") for c in components]

    resolved_cols: list[str] = []
    for canon in wanted_canon:
        if canon == "iter":
            continue
        actual = resolve_alias_from_columns(available, canon)
        if actual is None:
            raise KeyError(
                f"Component '{canon}' not found (and no alias matched). "
                f"Available: {', '.join(available)}"
            )
        if actual == "iter":
            continue
        resolved_cols.append(actual)

    rows: list[dict[str, object]] = []
    for _, src_row in df.iterrows():
        iter_val = int(src_row["iter"])
        for raw_col in resolved_cols:
            rows.append(
                {
                    "iter": iter_val,
                    "component": str(raw_col),
                    "value": src_row[raw_col],
                }
            )
    out = pd.DataFrame(rows, columns=["iter", "component", "value"])

    if include_geo_descriptor:
        out.insert(0, "geo_descriptor", str(data.geo_descriptor))
    return out


@dataclass
class GeometryOptimizationRequest(BaseRequest):
    """Request for selected fort.57 geometry-optimization components."""

    component: Sequence[str] | None = dc_field(
        default=None,
        metadata={
            "label": "Component",
            "help": "fort.57 components to include.",
            "choices": list(_GEOMETRY_OPT_COMPONENT_CHOICES),
        },
    )
    include_geo_descriptor: bool = dc_field(
        default=False,
        metadata={
            "label": "Include Geo Descriptor",
            "help": "Whether to include the geometry descriptor in output rows.",
            "choices": [True, False],
        },
    )


@dataclass
class GeometryOptimizationResult(BaseResult):
    """Geometry-optimization extraction result.

    Output structure:
    - request: GeometryOptimizationRequest used to generate this result
    - table: pandas.DataFrame with columns
      ['iter', 'component', 'value'] plus optional ['geo_descriptor']
      - iter: optimization iteration index from fort.57
      - component: selected component name ('E_pot', 'T', 'T_set', 'RMSG', 'nfc')
      - value: component value at that iteration
      - geo_descriptor: optional descriptor copied from parsed fort.57 metadata
    """

    table: pd.DataFrame
    request: GeometryOptimizationRequest


@register_task("geometry_optimization_data")
class GeometryOptimizationTask(AnalysisTask):
    """Return selected geometry-optimization summary data from fort.57."""

    required_data = GeometryOptimizationProgressData

    @staticmethod
    def recommended_presentations(
        _result: GeometryOptimizationResult, payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        table_rows = payload.get("table")
        if not isinstance(table_rows, list) or not table_rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = table_rows[0] if isinstance(table_rows[0], dict) else {}
        if "iter" not in sample or "value" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        group_by = "component" if "component" in sample else ""
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="RMSG vs Iter",
                mapping={
                    "x_col": "iter",
                    "y_col": "value",
                    "group_by_col": group_by,
                },
                options={
                    "title": "RMSG vs iter",
                    "xlabel": "iter",
                    "ylabel": "value",
                    "legend": bool(group_by),
                },
                view_type="plot2d",
            ),
        ]

    def run(
        self,
        data: GeometryOptimizationProgressData,
        request: GeometryOptimizationRequest,
        reporter=None,
    ) -> GeometryOptimizationResult:
        table = _build_geometry_optimization_table(
            data=data,
            components=request.component,
            include_geo_descriptor=bool(request.include_geo_descriptor),
        )
        return GeometryOptimizationResult(table=table, request=request)


__all__ = [
    "GeometryOptimizationRequest",
    "GeometryOptimizationResult",
    "GeometryOptimizationTask",
]
