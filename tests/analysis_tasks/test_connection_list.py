"""Sanity check for ConnectionListTask via AnalysisExecutor."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import reaxkit.engine  # noqa: F401 (register engine adapters)
from reaxkit.analysis.connectivity.connectivity import ConnectionListRequest, ConnectionListTask
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

    task = ConnectionListTask()
    task_name = str(task.__class__.__name__).replace("(", "").replace(")", "")
    task_artifacts_dir = ARTIFACTS_DIR / task_name
    task_artifacts_dir.mkdir(parents=True, exist_ok=True)
    request = ConnectionListRequest(
        frames=None,
        every=50,
        min_bo=0.3,
        undirected=True,
        include_self=False,
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
    assert {
        "frame_idx",
        "iteration",
        "source",
        "source_type",
        "destination",
        "destination_type",
        "BO",
    }.issubset(set(result.table.columns))

    payload = {"table": result.table.head(20).to_dict(orient="records")}
    views = task.recommended_presentations(result, payload)
    assert len(views) >= 2
    assert views[0].view_type == "table"
    assert views[1].view_type == "plot2d"

    metadata_path = task_artifacts_dir / "connection_list_summary.txt"
    csv_path = task_artifacts_dir / "connection_list.csv"
    head_path = task_artifacts_dir / "connection_list_head.txt"

    metadata_path.write_text(
        "\n".join(
            [
                f"Detected adapter: {adapter.__class__.__name__}",
                f"Result type: {type(result).__name__}",
                f"Columns: {list(result.table.columns)}",
                f"Rows: {len(result.table)}",
                f"Request every: {result.request.every}",
                f"Request min_bo: {result.request.min_bo}",
                f"Request undirected: {result.request.undirected}",
                f"Request include_self: {result.request.include_self}",
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


def test_connection_list_saves_artifacts() -> None:
    if not RUN_DIR.exists():
        pytest.skip(f"RUN_DIR does not exist: {RUN_DIR}")
    out_dir = _run_and_save()
    assert (out_dir / "connection_list_summary.txt").exists()
    assert (out_dir / "connection_list.csv").exists()
    assert (out_dir / "connection_list_head.txt").exists()


def main() -> None:
    if not RUN_DIR.exists():
        return
    _run_and_save()


if __name__ == "__main__":
    main()
