"""Parameter-optimization diagnostic analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ForceFieldOptimizationDiagnosticData


def _fort79_frame(data: ForceFieldOptimizationDiagnosticData) -> pd.DataFrame:
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


def _get_fort79_data_with_diff_sensitivities(
    data: ForceFieldOptimizationDiagnosticData,
) -> pd.DataFrame:
    """Compute relative force-field error sensitivities from fort.79 diagnostics."""
    df = _fort79_frame(data)
    diff3 = pd.to_numeric(df["diff3"], errors="coerce").replace(0.0, np.nan)

    result = pd.DataFrame({"identifier": df["identifier"].astype(object)})
    result["sensitivity1/3"] = pd.to_numeric(df["diff1"], errors="coerce") / diff3
    result["sensitivity2/3"] = pd.to_numeric(df["diff2"], errors="coerce") / diff3
    result["sensitivity4/3"] = pd.to_numeric(df["diff4"], errors="coerce") / diff3
    result["min_sensitivity"] = result[["sensitivity1/3", "sensitivity2/3", "sensitivity4/3"]].min(axis=1)
    result["max_sensitivity"] = result[["sensitivity1/3", "sensitivity2/3", "sensitivity4/3"]].max(axis=1)
    return result


@dataclass
class ParameterOptimizationDiagnosticRequest(BaseRequest):
    """Request for fort.79 parameter-optimization diagnostics."""


@dataclass
class ParameterOptimizationDiagnosticResult(BaseResult):
    """Result for fort.79 parameter-optimization diagnostics."""

    table: pd.DataFrame


@register_task("parameter_optimization_diagnostic")
class ParameterOptimizationDiagnosticTask(AnalysisTask):
    """Return sensitivity diagnostics derived from fort.79 parameter updates."""

    required_data = ForceFieldOptimizationDiagnosticData

    def run(
        self,
        data: ForceFieldOptimizationDiagnosticData,
        request: ParameterOptimizationDiagnosticRequest,
        reporter=None,
    ) -> ParameterOptimizationDiagnosticResult:
        table = _get_fort79_data_with_diff_sensitivities(data)
        return ParameterOptimizationDiagnosticResult(table=table)


__all__ = [
    "ParameterOptimizationDiagnosticRequest",
    "ParameterOptimizationDiagnosticResult",
    "ParameterOptimizationDiagnosticTask",
]
