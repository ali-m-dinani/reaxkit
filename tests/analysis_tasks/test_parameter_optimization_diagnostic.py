"""Sanity check for ParameterOptimizationDiagnosticTask via AnalysisExecutor."""

from __future__ import annotations

from pathlib import Path

import pytest
import reaxkit.engine  # noqa: F401 (register engine adapters)
from reaxkit.analysis.force_field.diagnostics import (
    ParameterOptimizationDiagnosticRequest,
    ParameterOptimizationDiagnosticTask,
)
from reaxkit.core.analysis_executor import AnalysisExecutor
from reaxkit.core.engine_registry import resolve_engine
from reaxkit.core.exceptions import AnalysisError

RUN_DIR = Path(
    r"C:\Users\alimo\PycharmProjects\pythonProject\reaxkit\examples_to_test"
)
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def _run_and_save(*, interpret: bool) -> Path:
    run_dir = RUN_DIR
    if not run_dir.exists():
        raise FileNotFoundError(f"RUN_DIR does not exist: {run_dir}")
    project_root = run_dir / "reaxkit_workspace"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    adapter = resolve_engine(str(run_dir), engine=None)

    task = ParameterOptimizationDiagnosticTask()
    task_name = task.__class__.__name__
    task_artifacts_dir = ARTIFACTS_DIR / task_name
    task_artifacts_dir.mkdir(parents=True, exist_ok=True)
    request = ParameterOptimizationDiagnosticRequest(interpret=interpret)
    executor = AnalysisExecutor()

    try:
        result = executor.run(
            task,
            request,
            {
                "run_dir": str(run_dir),
                "project_root": str(project_root),
                "cache": False,
            },
        )
    except AnalysisError as exc:
        msg = str(exc)
        if (
            "expected data type ForceFieldOptimizationDiagnosticBundleData, got NoneType" in msg
            or "Failed to load required data 'ForceFieldOptimizationDiagnosticBundleData'" in msg
        ):
            pytest.skip("ForceFieldOptimizationDiagnosticBundleData is not available for this run_dir.")
        raise

    assert result.request == request
    common_cols = {
        "value1",
        "value2",
        "value3",
        "diff1",
        "diff2",
        "diff3",
        "a",
        "b",
        "c",
        "parabol_min",
        "parabol_min_diff",
        "value4",
        "diff4",
        "sensitivity1/3",
        "sensitivity2/3",
        "sensitivity4/3",
        "min_sensitivity",
        "max_sensitivity",
    }
    assert common_cols.issubset(set(result.table.columns))
    if interpret:
        assert "identifier" not in result.table.columns
        assert {
            "ff_section",
            "ff_section_line",
            "ff_parameter",
            "component",
            "ffield_section_name",
            "ffield_value",
            "term",
        }.issubset(set(result.table.columns))
    else:
        assert "identifier" in result.table.columns

    suffix = "interpreted" if interpret else "raw"
    metadata_path = task_artifacts_dir / f"parameter_optimization_diagnostic_summary_{suffix}.txt"
    csv_path = task_artifacts_dir / f"parameter_optimization_diagnostic_{suffix}.csv"
    head_path = task_artifacts_dir / f"parameter_optimization_diagnostic_head_{suffix}.txt"

    metadata_path.write_text(
        "\n".join(
            [
                f"Detected adapter: {adapter.__class__.__name__}",
                f"Result type: {type(result).__name__}",
                f"Interpret: {interpret}",
                f"Columns: {list(result.table.columns)}",
                f"Rows: {len(result.table)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        result.table.to_csv(csv_path, index=False)
        head_path.write_text(result.table.head(12).to_string(index=False) + "\n", encoding="utf-8")
    except PermissionError as exc:
        pytest.skip(f"Artifact file is locked by another process: {exc}")
    return task_artifacts_dir


@pytest.mark.parametrize("interpret", [False, True])
def test_parameter_optimization_diagnostic_saves_artifacts(interpret: bool) -> None:
    if not RUN_DIR.exists():
        pytest.skip(f"RUN_DIR does not exist: {RUN_DIR}")
    out_dir = _run_and_save(interpret=interpret)
    suffix = "interpreted" if interpret else "raw"
    assert (out_dir / f"parameter_optimization_diagnostic_summary_{suffix}.txt").exists()
    assert (out_dir / f"parameter_optimization_diagnostic_{suffix}.csv").exists()
    assert (out_dir / f"parameter_optimization_diagnostic_head_{suffix}.txt").exists()


def main() -> None:
    if not RUN_DIR.exists():
        return
    _run_and_save(interpret=False)
    _run_and_save(interpret=True)


if __name__ == "__main__":
    main()
