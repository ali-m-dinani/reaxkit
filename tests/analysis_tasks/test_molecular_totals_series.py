"""Sanity check for MolecularTotalsSeriesTask via AnalysisExecutor."""

from __future__ import annotations

from pathlib import Path

import pytest
import reaxkit.engine  # noqa: F401 (register engine adapters)
from reaxkit.analysis.timeseries.timeseries import MolecularTotalsSeriesRequest, MolecularTotalsSeriesTask
from reaxkit.core.analysis_executor import AnalysisExecutor
from reaxkit.core.exceptions import AnalysisError
from reaxkit.core.engine_registry import resolve_engine

RUN_DIR = Path(
    r"C:\Users\alimo\PycharmProjects\pythonProject\reaxkit\examples_to_test"
)
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def _run_and_save() -> Path:
    run_dir = RUN_DIR
    if not run_dir.exists():
        raise FileNotFoundError(f"RUN_DIR does not exist: {run_dir}")
    project_root = run_dir / "reaxkit_workspace"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    adapter = resolve_engine(str(run_dir), engine=None)

    task = MolecularTotalsSeriesTask()
    task_name = str(task.__class__.__name__).replace("(", "").replace(")", "")
    task_artifacts_dir = ARTIFACTS_DIR / task_name
    task_artifacts_dir.mkdir(parents=True, exist_ok=True)
    request = MolecularTotalsSeriesRequest(
        quantities=["total_molecules", "total_atoms"],
        every=1,
    )
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
        if "expected data type MolecularAnalysisData, got NoneType" in msg or "Failed to load required data 'MolecularAnalysisData'" in msg:
            pytest.skip("MolecularAnalysisData is not available for this run_dir.")
        raise

    assert result.request == request
    assert {"frame_index", "iter", "quantity", "value"}.issubset(set(result.table.columns))

    metadata_path = task_artifacts_dir / "molecular_totals_series_summary.txt"
    csv_path = task_artifacts_dir / "molecular_totals_series.csv"
    head_path = task_artifacts_dir / "molecular_totals_series_head.txt"

    metadata_path.write_text(
        "\n".join(
            [
                f"Detected adapter: {adapter.__class__.__name__}",
                f"Result type: {type(result).__name__}",
                f"Columns: {list(result.table.columns)}",
                f"Rows: {len(result.table)}",
                f"Request quantities: {list(result.request.quantities)}",
                f"Request frames: {list(result.request.frames) if result.request.frames is not None else None}",
                f"Request every: {result.request.every}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result.table.to_csv(csv_path, index=False)
    head_path.write_text(result.table.head(12).to_string(index=False) + "\n", encoding="utf-8")
    return task_artifacts_dir


def test_molecular_totals_series_saves_artifacts() -> None:
    if not RUN_DIR.exists():
        pytest.skip(f"RUN_DIR does not exist: {RUN_DIR}")
    out_dir = _run_and_save()
    assert (out_dir / "molecular_totals_series_summary.txt").exists()
    assert (out_dir / "molecular_totals_series.csv").exists()
    assert (out_dir / "molecular_totals_series_head.txt").exists()


def main() -> None:
    if not RUN_DIR.exists():
        return
    _run_and_save()


if __name__ == "__main__":
    main()
