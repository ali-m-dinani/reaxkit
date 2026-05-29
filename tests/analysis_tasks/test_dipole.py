"""Sanity check for DipoleTask via AnalysisExecutor."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import reaxkit.engine  # noqa: F401 (register engine adapters)
from reaxkit.analysis.electrostatics.electrostatics import DipoleRequest, DipoleTask
from reaxkit.core.platform.engine_resolver import resolve_engine
from reaxkit.domain.data_models import ChargeData, ConnectivityData, ElectrostaticsData, TrajectoryData

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

    task = DipoleTask()
    task_name = str(task.__class__.__name__).replace("(", "").replace(")", "")
    task_artifacts_dir = ARTIFACTS_DIR / task_name
    task_artifacts_dir.mkdir(parents=True, exist_ok=True)
    request = DipoleRequest(
        scope="total",
        frames=[0],
        every=1,
    )
    load_args = {
        "run_dir": str(run_dir),
        "project_root": str(project_root),
        "xmolout": str(run_dir / "xmolout"),
        "fort7": str(run_dir / "fort.7"),
        "cache": False,
    }
    data = ElectrostaticsData(
        trajectory=adapter.load(TrajectoryData, load_args),
        charges=adapter.load(ChargeData, load_args),
        connectivity=adapter.load(ConnectivityData, load_args),
    )
    result = task.run(data, request)
    assert result.request == request
    assert isinstance(result.table, pd.DataFrame)
    assert {"frame_index", "iter", "mu_x (debye)", "mu_y (debye)", "mu_z (debye)"}.issubset(set(result.table.columns))

    payload = {"table": result.table.head(20).to_dict(orient="records")}
    views = task.recommended_presentations(result, payload)
    assert views[0].view_type == "table"
    if payload["table"]:
        assert len(views) >= 2
        assert views[1].view_type == "plot2d"

    metadata_path = task_artifacts_dir / "dipole_summary.txt"
    csv_path = task_artifacts_dir / "dipole.csv"
    head_path = task_artifacts_dir / "dipole_head.txt"

    metadata_path.write_text(
        "\n".join(
            [
                f"Detected adapter: {adapter.__class__.__name__}",
                f"Result type: {type(result).__name__}",
                f"Columns: {list(result.table.columns)}",
                f"Rows: {len(result.table)}",
                f"Request scope: {result.request.scope}",
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


def test_dipole_saves_artifacts() -> None:
    if not RUN_DIR.exists():
        pytest.skip(f"RUN_DIR does not exist: {RUN_DIR}")
    out_dir = _run_and_save()
    assert (out_dir / "dipole_summary.txt").exists()
    assert (out_dir / "dipole.csv").exists()
    assert (out_dir / "dipole_head.txt").exists()


def main() -> None:
    if not RUN_DIR.exists():
        return
    _run_and_save()


if __name__ == "__main__":
    main()
