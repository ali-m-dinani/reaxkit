"""Sanity check for TrajectoryRelabelByCoordinationTask via AnalysisExecutor."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import reaxkit.engine  # noqa: F401 (register engine adapters)
from reaxkit.analysis.trajectory.relabel import (
    TrajectoryRelabelByCoordinationRequest,
    TrajectoryRelabelByCoordinationTask,
)
from reaxkit.core.analysis_executor import AnalysisExecutor
from reaxkit.core.engine_registry import resolve_engine
from reaxkit.core.exceptions import AnalysisError

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

    task = TrajectoryRelabelByCoordinationTask()
    task_name = str(task.__class__.__name__).replace("(", "").replace(")", "")
    task_artifacts_dir = ARTIFACTS_DIR / task_name
    task_artifacts_dir.mkdir(parents=True, exist_ok=True)
    request = TrajectoryRelabelByCoordinationRequest(
        labels={-1: "U", 0: "C", 1: "O"},
        mode="by_type",
        keep_coord_original=True,
        frames=[0],
        every=1,
        valences=None,
        threshold=0.9,
        require_all_valences=False,
    )
    executor = AnalysisExecutor()

    try:
        result = executor.run(
            task,
            request,
            {
                "run_dir": str(run_dir),
                "project_root": str(project_root),
                "xmolout": str(run_dir / "xmolout"),
                "fort7": str(run_dir / "fort.7"),
                "ffield": str(run_dir / "ffield"),
                "cache": False,
            },
        )
    except AnalysisError as exc:
        msg = str(exc)
        if "force-field valences" in msg or "Failed to parse FFieldHandler" in msg:
            pytest.skip("Force-field data is not available for trajectory relabel by coordination.")
        raise

    assert result.request == request
    assert isinstance(result.table, pd.DataFrame)
    assert {"frame_index", "iter", "atom_id", "atom_type", "status"}.issubset(set(result.table.columns))
    assert result.trajectory.atom_labels is not None
    assert result.trajectory.atom_labels.shape[0] >= 1

    payload = {"table": result.table.head(20).to_dict(orient="records")}
    views = task.recommended_presentations(result, payload)
    assert len(views) == 1
    assert views[0].view_type == "table"

    metadata_path = task_artifacts_dir / "trajectory_relabel_by_coordination_summary.txt"
    csv_path = task_artifacts_dir / "trajectory_relabel_by_coordination.csv"
    head_path = task_artifacts_dir / "trajectory_relabel_by_coordination_head.txt"

    metadata_path.write_text(
        "\n".join(
            [
                f"Detected adapter: {adapter.__class__.__name__}",
                f"Result type: {type(result).__name__}",
                f"Columns: {list(result.table.columns)}",
                f"Rows: {len(result.table)}",
                f"Request mode: {result.request.mode}",
                f"Request keep_coord_original: {result.request.keep_coord_original}",
                f"Trajectory frames: {result.trajectory.positions.shape[0]}",
                f"Trajectory atoms: {result.trajectory.positions.shape[1]}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result.table.to_csv(csv_path, index=False)
    head_path.write_text(result.table.head(12).to_string(index=False) + "\n", encoding="utf-8")
    return task_artifacts_dir


def test_trajectory_relabel_by_coordination_saves_artifacts() -> None:
    if not RUN_DIR.exists():
        pytest.skip(f"RUN_DIR does not exist: {RUN_DIR}")
    out_dir = _run_and_save()
    assert (out_dir / "trajectory_relabel_by_coordination_summary.txt").exists()
    assert (out_dir / "trajectory_relabel_by_coordination.csv").exists()
    assert (out_dir / "trajectory_relabel_by_coordination_head.txt").exists()


def main() -> None:
    if not RUN_DIR.exists():
        return
    _run_and_save()


if __name__ == "__main__":
    main()
