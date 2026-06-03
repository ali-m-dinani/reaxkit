"""Sanity check for ConnectionTableTask via AnalysisExecutor."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import reaxkit.engine  # noqa: F401 (register engine adapters)
from reaxkit.analysis.connectivity.connectivity import ConnectionTableRequest, ConnectionTableTask
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.core.platform.engine_resolver import resolve_engine
from reaxkit.core.platform.exceptions import AnalysisError

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

    task = ConnectionTableTask()
    task_name = str(task.__class__.__name__).replace("(", "").replace(")", "")
    task_artifacts_dir = ARTIFACTS_DIR / task_name
    task_artifacts_dir.mkdir(parents=True, exist_ok=True)
    request = ConnectionTableRequest(
        frame=0,
        min_bo=0.0,
        undirected=True,
        fill_value=0.0,
    )
    executor = AnalysisExecutor()

    try:
        result = executor.run(
            task,
            request,
            {
                "run_dir": str(run_dir),
                "project_root": str(project_root),
                "fort7": str(run_dir / "fort.7"),
                "xmolout": str(run_dir / "xmolout"),
                "engine": "reaxff",
                "cache": False,
            },
        )
    except AnalysisError as exc:
        msg = str(exc)
        if (
            "expected data type ConnectivityData, got NoneType" in msg
            or "Failed to load required data 'ConnectivityData'" in msg
        ):
            pytest.skip("ConnectivityData is not available for this run_dir.")
        raise

    assert result.request == request
    assert isinstance(result.table, pd.DataFrame)

    table_for_views = result.table.reset_index() if not isinstance(result.table.index, pd.RangeIndex) else result.table
    payload = {"table": table_for_views.head(20).to_dict(orient="records")}
    views = task.recommended_presentations(result, payload)
    assert len(views) >= 2
    assert views[0].view_type == "table"
    assert views[1].view_type == "plot2d"

    metadata_path = task_artifacts_dir / "connection_table_summary.txt"
    csv_path = task_artifacts_dir / "connection_table.csv"
    head_path = task_artifacts_dir / "connection_table_head.txt"

    metadata_path.write_text(
        "\n".join(
            [
                f"Detected adapter: {adapter.__class__.__name__}",
                f"Result type: {type(result).__name__}",
                f"Columns: {list(result.table.columns)}",
                f"Rows: {len(result.table)}",
                f"Request frame: {result.request.frame}",
                f"Request min_bo: {result.request.min_bo}",
                f"Request undirected: {result.request.undirected}",
                f"Request fill_value: {result.request.fill_value}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        result.table.to_csv(csv_path, index=True)
        head_path.write_text(result.table.head(12).to_string(index=True) + "\n", encoding="utf-8")
    except PermissionError as exc:
        pytest.skip(f"Artifact file is locked by another process: {exc}")
    return task_artifacts_dir


def test_connection_table_saves_artifacts() -> None:
    if not RUN_DIR.exists():
        pytest.skip(f"RUN_DIR does not exist: {RUN_DIR}")
    out_dir = _run_and_save()
    assert (out_dir / "connection_table_summary.txt").exists()
    assert (out_dir / "connection_table.csv").exists()
    assert (out_dir / "connection_table_head.txt").exists()


def main() -> None:
    if not RUN_DIR.exists():
        return
    _run_and_save()


if __name__ == "__main__":
    main()
