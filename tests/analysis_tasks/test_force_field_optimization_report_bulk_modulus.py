"""Sanity check for ForceFieldOptimizationReportBulkModulusTask via AnalysisExecutor."""

from __future__ import annotations

from pathlib import Path

import pytest
import reaxkit.engine  # noqa: F401 (register engine adapters)
from reaxkit.analysis.force_field.report import (
    ForceFieldOptimizationReportBulkModulusRequest,
    ForceFieldOptimizationReportBulkModulusTask,
)
from reaxkit.core.analysis_executor import AnalysisExecutor
from reaxkit.core.engine_registry import resolve_engine
from reaxkit.core.exceptions import AnalysisError

RUN_DIR = Path(
    r"C:\Users\alimo\PycharmProjects\pythonProject\reaxkit\examples_to_test"
)
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifcats"


def _run_and_save() -> Path:
    run_dir = RUN_DIR
    if not run_dir.exists():
        raise FileNotFoundError(f"RUN_DIR does not exist: {run_dir}")
    project_root = run_dir / "reaxkit_workspace"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    adapter = resolve_engine(str(run_dir), engine=None)

    task = ForceFieldOptimizationReportBulkModulusTask()
    task_name = str(task.__class__.__name__).replace("(", "").replace(")", "")
    task_artifacts_dir = ARTIFACTS_DIR / task_name
    task_artifacts_dir.mkdir(parents=True, exist_ok=True)
    request = ForceFieldOptimizationReportBulkModulusRequest(
        iden="all",
        shift_min_to_zero=True,
        flip_sign=False,
        min_points=6,
    )
    executor = AnalysisExecutor()

    try:
        result = executor.run(
            task,
            request,
            {
                "run_dir": str(run_dir),
                "project_root": str(project_root),
                "fort99": str(run_dir / "fort.99"),
                "fort74": str(run_dir / "fort.74"),
                "engine": "reaxff",
                "cache": False,
            },
        )
    except AnalysisError as exc:
        msg = str(exc)
        if (
            "expected data type ForceFieldOptimizationReportEOSBundleData, got NoneType" in msg
            or "Failed to load required data 'ForceFieldOptimizationReportEOSBundleData'" in msg
        ):
            pytest.skip("ForceFieldOptimizationReportEOSBundleData is not available for this run_dir.")
        raise

    assert result.request == request
    assert {
        "base_iden",
        "n_points",
        "V0_A3",
        "K0_eV_A3",
        "K0_GPa",
        "E0_eV",
        "C",
        "success",
    }.issubset(set(result.table.columns))

    metadata_path = task_artifacts_dir / "force_field_optimization_report_bulk_modulus_summary.txt"
    csv_path = task_artifacts_dir / "force_field_optimization_report_bulk_modulus.csv"
    head_path = task_artifacts_dir / "force_field_optimization_report_bulk_modulus_head.txt"

    metadata_path.write_text(
        "\n".join(
            [
                f"Detected adapter: {adapter.__class__.__name__}",
                f"Result type: {type(result).__name__}",
                f"Columns: {list(result.table.columns)}",
                f"Rows: {len(result.table)}",
                f"Request iden: {result.request.iden}",
                f"Request shift_min_to_zero: {result.request.shift_min_to_zero}",
                f"Request flip_sign: {result.request.flip_sign}",
                f"Request min_points: {result.request.min_points}",
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


def test_force_field_optimization_report_bulk_modulus_saves_artifacts() -> None:
    if not RUN_DIR.exists():
        pytest.skip(f"RUN_DIR does not exist: {RUN_DIR}")
    out_dir = _run_and_save()
    assert (out_dir / "force_field_optimization_report_bulk_modulus_summary.txt").exists()
    assert (out_dir / "force_field_optimization_report_bulk_modulus.csv").exists()
    assert (out_dir / "force_field_optimization_report_bulk_modulus_head.txt").exists()


def main() -> None:
    if not RUN_DIR.exists():
        return
    _run_and_save()


if __name__ == "__main__":
    main()
