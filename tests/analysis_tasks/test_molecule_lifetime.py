"""Sanity check for MoleculeLifetimeTask via AnalysisExecutor."""

from __future__ import annotations

from pathlib import Path

import pytest
import reaxkit.engine  # noqa: F401 (register engine adapters)
from reaxkit.analysis.molecular_analysis.molecular_analysis import (
    MoleculeLifetimeRequest,
    MoleculeLifetimeTask,
)
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

    task = MoleculeLifetimeTask()
    task_name = task.__class__.__name__
    task_artifacts_dir = ARTIFACTS_DIR / task_name
    task_artifacts_dir.mkdir(parents=True, exist_ok=True)
    request = MoleculeLifetimeRequest(
        every=1,
        min_freq=1.0,
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
        if (
            "expected data type MolecularAnalysisData, got NoneType" in msg
            or "Failed to load required data 'MolecularAnalysisData'" in msg
        ):
            pytest.skip("MolecularAnalysisData is not available for this run_dir.")
        raise

    assert result.request == request
    assert {
        "molecular_formula",
        "lifetime_segment_id",
        "start_frame_index",
        "end_frame_index",
        "start_iter",
        "end_iter",
        "lifetime_segment_sampled_step_count",
        "peak_freq",
        "mean_freq",
    }.issubset(set(result.table.columns))

    metadata_path = task_artifacts_dir / "molecule_lifetime_summary.txt"
    csv_path = task_artifacts_dir / "molecule_lifetime.csv"
    head_path = task_artifacts_dir / "molecule_lifetime_head.txt"

    metadata_path.write_text(
        "\n".join(
            [
                f"Detected adapter: {adapter.__class__.__name__}",
                f"Result type: {type(result).__name__}",
                f"Columns: {list(result.table.columns)}",
                f"Rows: {len(result.table)}",
                f"Request molecules: {list(result.request.molecules) if result.request.molecules is not None else None}",
                f"Request frames: {list(result.request.frames) if result.request.frames is not None else None}",
                f"Request every: {result.request.every}",
                f"Request min_freq: {result.request.min_freq}",
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


def test_molecule_lifetime_saves_artifacts() -> None:
    if not RUN_DIR.exists():
        pytest.skip(f"RUN_DIR does not exist: {RUN_DIR}")
    out_dir = _run_and_save()
    assert (out_dir / "molecule_lifetime_summary.txt").exists()
    assert (out_dir / "molecule_lifetime.csv").exists()
    assert (out_dir / "molecule_lifetime_head.txt").exists()


def main() -> None:
    if not RUN_DIR.exists():
        return
    _run_and_save()


if __name__ == "__main__":
    main()
