"""Sanity check for ForceFieldOptimizationParameterTask via AnalysisExecutor."""

from __future__ import annotations

from pathlib import Path

import pytest
import reaxkit.engine  # noqa: F401 (register engine adapters)
from reaxkit.analysis.params.params import (
    ForceFieldOptimizationParameterRequest,
    ForceFieldOptimizationParameterTask,
)
from reaxkit.core.analysis_executor import AnalysisExecutor
from reaxkit.core.engine_registry import resolve_engine
from reaxkit.core.exceptions import AnalysisError

RUN_DIR = Path(
    r"C:\Users\alimo\PycharmProjects\pythonProject\reaxkit\examples_to_test"
)
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
SUMMARY_FILE = "force_field_optimization_parameter_summary.txt"
CSV_FILE = "force_field_optimization_parameter.csv"
HEAD_FILE = "force_field_optimization_parameter_head.txt"
REQUIRED_COLUMNS = {"component", "ff_section", "ff_section_line", "ff_parameter", "search_interval"}


def _run_and_save() -> Path:
    run_dir = RUN_DIR
    if not run_dir.exists():
        raise FileNotFoundError(f"RUN_DIR does not exist: {run_dir}")
    project_root = run_dir / "reaxkit_workspace"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    adapter = resolve_engine(str(run_dir), engine=None)

    task = ForceFieldOptimizationParameterTask()
    task_name = task.__class__.__name__
    task_artifacts_dir = ARTIFACTS_DIR / task_name
    task_artifacts_dir.mkdir(parents=True, exist_ok=True)
    request = ForceFieldOptimizationParameterRequest(
        drop_duplicate=True,
        interpret=True,
    )
    executor = AnalysisExecutor()

    try:
        analysis_args = {
            "run_dir": str(run_dir),
            "project_root": str(project_root),
            "ffield": str(run_dir / "ffield"),
            "params": str(run_dir / "params"),
            "engine": "reaxff",
            "cache": False,
        }
        result = executor.run(task, request, analysis_args)
    except AnalysisError as exc:
        msg = str(exc)
        if (
            "expected data type ForceFieldOptimizationParameterBundleData, got NoneType" in msg
            or "Failed to load required data 'ForceFieldOptimizationParameterBundleData'" in msg
        ):
            pytest.skip("ForceFieldOptimizationParameterBundleData is not available for this run_dir.")
        raise

    assert result.request == request
    assert REQUIRED_COLUMNS.issubset(set(result.table.columns))

    metadata_path = task_artifacts_dir / SUMMARY_FILE
    csv_path = task_artifacts_dir / CSV_FILE
    head_path = task_artifacts_dir / HEAD_FILE

    metadata_path.write_text(
        "\n".join(
            [
                f"Detected adapter: {adapter.__class__.__name__}",
                f"Result type: {type(result).__name__}",
                f"Columns: {list(result.table.columns)}",
                f"Rows: {len(result.table)}",
                f"Request drop_duplicate: {result.request.drop_duplicate}",
                f"Request interpret: {result.request.interpret}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result.table.to_csv(csv_path, index=False)
    head_path.write_text(result.table.head(12).to_string(index=False) + "\n", encoding="utf-8")
    return task_artifacts_dir


def test_force_field_optimization_parameter_saves_artifacts() -> None:
    if not RUN_DIR.exists():
        pytest.skip(f"RUN_DIR does not exist: {RUN_DIR}")
    out_dir = _run_and_save()
    assert (out_dir / SUMMARY_FILE).exists()
    assert (out_dir / CSV_FILE).exists()
    assert (out_dir / HEAD_FILE).exists()


def main() -> None:
    if not RUN_DIR.exists():
        return
    _run_and_save()


if __name__ == "__main__":
    main()
