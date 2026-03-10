"""Sanity check for LargestMoleculeCompositionTask via AnalysisExecutor."""

from __future__ import annotations

from pathlib import Path

import pytest
import reaxkit.engine  # noqa: F401 (register engine adapters)
from reaxkit.analysis.molecular_analysis.molecular_analysis import (
    LargestMoleculeCompositionRequest,
    LargestMoleculeCompositionTask,
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

    task = LargestMoleculeCompositionTask()
    task_name = task.__class__.__name__
    task_artifacts_dir = ARTIFACTS_DIR / task_name
    task_artifacts_dir.mkdir(parents=True, exist_ok=True)
    request = LargestMoleculeCompositionRequest(
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
        if (
            "expected data type MolecularAnalysisData, got NoneType" in msg
            or "Failed to load required data 'MolecularAnalysisData'" in msg
        ):
            pytest.skip("MolecularAnalysisData is not available for this run_dir.")
        raise

    assert result.request == request
    assert {"frame_index", "iter", "element", "count"}.issubset(set(result.table.columns))

    metadata_path = task_artifacts_dir / "largest_molecule_composition_summary.txt"
    csv_path = task_artifacts_dir / "largest_molecule_composition.csv"
    head_path = task_artifacts_dir / "largest_molecule_composition_head.txt"

    metadata_path.write_text(
        "\n".join(
            [
                f"Detected adapter: {adapter.__class__.__name__}",
                f"Result type: {type(result).__name__}",
                f"Columns: {list(result.table.columns)}",
                f"Rows: {len(result.table)}",
                f"Request frames: {list(result.request.frames) if result.request.frames is not None else None}",
                f"Request every: {result.request.every}",
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


def test_largest_molecule_composition_saves_artifacts() -> None:
    if not RUN_DIR.exists():
        pytest.skip(f"RUN_DIR does not exist: {RUN_DIR}")
    out_dir = _run_and_save()
    assert (out_dir / "largest_molecule_composition_summary.txt").exists()
    assert (out_dir / "largest_molecule_composition.csv").exists()
    assert (out_dir / "largest_molecule_composition_head.txt").exists()


def main() -> None:
    if not RUN_DIR.exists():
        return
    _run_and_save()


if __name__ == "__main__":
    main()
