"""Sanity check for RDFTask via AnalysisExecutor."""

from __future__ import annotations

from pathlib import Path

import pytest
import reaxkit.engine  # noqa: F401 (register engine adapters)
from reaxkit.analysis.trajectory.rdf import RDFRequest, RDFTask
from reaxkit.core.analysis_executor import AnalysisExecutor
from reaxkit.core.engine_registry import resolve_engine

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

    task = RDFTask()
    task_name = str(task.__class__.__name__).replace("(", "").replace(")", "")
    task_artifacts_dir = ARTIFACTS_DIR / task_name
    task_artifacts_dir.mkdir(parents=True, exist_ok=True)
    request = RDFRequest(
        atom_ids_a=[1],
        atom_ids_b=[1],
        bins=100,
        every=1,
    )
    executor = AnalysisExecutor()

    result = executor.run(
        task,
        request,
        {
            "run_dir": str(run_dir),
            "project_root": str(project_root),
            "cache": False,
        },
    )
    assert result.request == request

    metadata_path = task_artifacts_dir / "trajectory_rdf_summary.txt"
    csv_path = task_artifacts_dir / "trajectory_rdf.csv"
    head_path = task_artifacts_dir / "trajectory_rdf_head.txt"

    metadata_path.write_text(
        "\n".join(
            [
                f"Detected adapter: {adapter.__class__.__name__}",
                f"Result type: {type(result).__name__}",
                f"Columns: {list(result.table.columns)}",
                f"Rows: {len(result.table)}",
                f"Request atom_ids_a: {list(result.request.atom_ids_a) if result.request.atom_ids_a is not None else None}",
                f"Request atom_ids_b: {list(result.request.atom_ids_b) if result.request.atom_ids_b is not None else None}",
                f"Request bins: {result.request.bins}",
                f"Request every: {result.request.every}",
                f"Request backend: {result.request.backend}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result.table.to_csv(csv_path, index=False)
    head_path.write_text(result.table.head(12).to_string(index=False) + "\n", encoding="utf-8")
    return task_artifacts_dir


def test_trajectory_rdf_saves_artifacts() -> None:
    if not RUN_DIR.exists():
        pytest.skip(f"RUN_DIR does not exist: {RUN_DIR}")
    out_dir = _run_and_save()
    assert (out_dir / "trajectory_rdf_summary.txt").exists()
    assert (out_dir / "trajectory_rdf.csv").exists()
    assert (out_dir / "trajectory_rdf_head.txt").exists()


def main() -> None:
    if not RUN_DIR.exists():
        return
    _run_and_save()


if __name__ == "__main__":
    main()
