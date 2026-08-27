"""Sanity check for GetTrainsetDataTask via AnalysisExecutor."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import reaxkit.engine  # noqa: F401 (register engine adapters)
from reaxkit.analysis.force_field.trainset import TrainsetDataRequest, TrainsetDataTask
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.core.platform.engine_resolver import resolve_engine
from reaxkit.core.platform.exceptions import AnalysisError
from reaxkit.domain.data_models import ForceFieldOptimizationTrainingSetData

RUN_DIR = Path(
    r"C:\Users\alimo\PycharmProjects\pythonProject\reaxkit\examples_to_test"
)
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def test_trainset_data_result_keeps_native_section_tables() -> None:
    data = ForceFieldOptimizationTrainingSetData(
        charge=pd.DataFrame({"atom": [1], "charge": [-0.5]}),
        heatfo=pd.DataFrame({"structure": ["bulk"], "heatfo": [-1.2]}),
        geometry=pd.DataFrame({"atom1": [1], "atom2": [2], "distance": [1.9]}),
        cell_parameters=pd.DataFrame({"structure": ["bulk"], "a": [3.1]}),
        energy=pd.DataFrame({"id1": ["bulk"], "lit": [-15.4]}),
    )

    result = TrainsetDataTask().run(data, TrainsetDataRequest(section="all"))

    assert list(result.section_tables) == [
        "CHARGE",
        "HEATFO",
        "GEOMETRY",
        "CELL_PARAMETERS",
        "ENERGY",
    ]
    assert list(result.section_tables["CHARGE"].columns) == ["atom", "charge"]
    assert "lit" not in result.section_tables["CHARGE"].columns
    assert list(result.section_tables["ENERGY"].columns) == ["id1", "lit"]


def test_trainset_data_result_limits_section_tables_to_selection() -> None:
    data = ForceFieldOptimizationTrainingSetData(
        charge=pd.DataFrame(),
        heatfo=pd.DataFrame(),
        geometry=pd.DataFrame({"distance": [1.9]}),
        cell_parameters=pd.DataFrame(),
        energy=pd.DataFrame(),
    )

    result = TrainsetDataTask().run(data, TrainsetDataRequest(section="geometry"))

    assert list(result.section_tables) == ["GEOMETRY"]
    assert result.section_tables["GEOMETRY"].to_dict(orient="records") == [
        {"distance": 1.9}
    ]


def _run_and_save() -> Path:
    run_dir = RUN_DIR
    if not run_dir.exists():
        raise FileNotFoundError(f"RUN_DIR does not exist: {run_dir}")
    project_root = run_dir / "reaxkit_workspace"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    adapter = resolve_engine(str(run_dir), engine=None)

    task = TrainsetDataTask()
    task_name = str(task.__class__.__name__).replace("(", "").replace(")", "")
    task_artifacts_dir = ARTIFACTS_DIR / task_name
    task_artifacts_dir.mkdir(parents=True, exist_ok=True)
    request = TrainsetDataRequest(section="all")
    executor = AnalysisExecutor()

    try:
        result = executor.run(
            task,
            request,
            {
                "run_dir": str(run_dir),
                "project_root": str(project_root),
                "trainset": str(run_dir / "trainset.in"),
                "engine": "reaxff",
                "cache": False,
            },
        )
    except AnalysisError as exc:
        msg = str(exc)
        if (
            "expected data type ForceFieldOptimizationTrainingSetData, got NoneType" in msg
            or "Failed to load required data 'ForceFieldOptimizationTrainingSetData'" in msg
        ):
            pytest.skip("ForceFieldOptimizationTrainingSetData is not available for this run_dir.")
        raise

    assert result.request == request
    assert not result.table.empty
    assert "section" in result.table.columns
    assert "line_number" in result.table.columns

    payload = {"table": result.table.head(20).to_dict(orient="records")}
    views = task.recommended_presentations(result, payload)
    assert len(views) >= 2
    assert views[0].view_type == "table"
    assert views[1].view_type == "plot2d"

    metadata_path = task_artifacts_dir / "trainset_data_summary.txt"
    csv_path = task_artifacts_dir / "trainset_data.csv"
    head_path = task_artifacts_dir / "trainset_data_head.txt"

    metadata_path.write_text(
        "\n".join(
            [
                f"Detected adapter: {adapter.__class__.__name__}",
                f"Result type: {type(result).__name__}",
                f"Columns: {list(result.table.columns)}",
                f"Rows: {len(result.table)}",
                f"Request section: {result.request.section}",
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


def test_trainset_data_saves_artifacts() -> None:
    if not RUN_DIR.exists():
        pytest.skip(f"RUN_DIR does not exist: {RUN_DIR}")
    out_dir = _run_and_save()
    assert (out_dir / "trainset_data_summary.txt").exists()
    assert (out_dir / "trainset_data.csv").exists()
    assert (out_dir / "trainset_data_head.txt").exists()


def main() -> None:
    if not RUN_DIR.exists():
        return
    _run_and_save()


if __name__ == "__main__":
    main()
