"""Geometry-optimization data extraction tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.alias import normalize_choice, resolve_alias_from_columns
from reaxkit.core.task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import GeometryOptimizationProgressData

_F57_CANONICAL = ("iter", "E_pot", "T", "T_set", "RMSG", "nfc")


def _fort57_frame(data: GeometryOptimizationProgressData) -> pd.DataFrame:
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


def _get_fort57_data(
    *,
    data: GeometryOptimizationProgressData,
    cols: Sequence[str] | None = None,
    include_geo_descriptor: bool = False,
) -> pd.DataFrame:
    """Extract selected columns from ``fort.57`` data as a DataFrame."""
    df = _fort57_frame(data)

    if cols is None or len(cols) == 0:
        out = df.copy()
    else:
        wanted_canon = [normalize_choice(c, domain="fort57.md") for c in cols]
        resolved_cols: list[str] = []
        available = list(df.columns)

        for canon in wanted_canon:
            if canon not in _F57_CANONICAL:
                raise ValueError(
                    f"Unknown fort.57 column '{canon}'. "
                    f"Allowed: {', '.join(_F57_CANONICAL)}"
                )

            actual = resolve_alias_from_columns(available, canon)
            if actual is None:
                raise KeyError(
                    f"Column '{canon}' not found (and no alias matched). "
                    f"Available: {', '.join(available)}"
                )
            resolved_cols.append(actual)

        out = df.loc[:, resolved_cols].copy()
        out = out.rename(columns=dict(zip(resolved_cols, wanted_canon)))

    if include_geo_descriptor:
        out.insert(0, "geo_descriptor", str(data.geo_descriptor))

    return out


@dataclass
class GeometryOptimizationRequest(BaseRequest):
    """Request for selected fort.57 geometry-optimization columns."""

    cols: Sequence[str] | None = None
    include_geo_descriptor: bool = False


@dataclass
class GeometryOptimizationResult(BaseResult):
    """Result for selected fort.57 geometry-optimization columns."""

    table: pd.DataFrame


@register_task("geometry_optimization_data")
class GeometryOptimizationTask(AnalysisTask):
    """Return selected geometry-optimization summary data from fort.57."""

    required_data = GeometryOptimizationProgressData

    def run(
        self,
        data: GeometryOptimizationProgressData,
        request: GeometryOptimizationRequest,
        reporter=None,
    ) -> GeometryOptimizationResult:
        table = _get_fort57_data(
            data=data,
            cols=request.cols,
            include_geo_descriptor=bool(request.include_geo_descriptor),
        )
        return GeometryOptimizationResult(table=table)


__all__ = [
    "GeometryOptimizationRequest",
    "GeometryOptimizationResult",
    "GeometryOptimizationTask",
]
