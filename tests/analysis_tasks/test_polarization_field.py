"""Sanity check for PolarizationFieldTask."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
import reaxkit.engine  # noqa: F401 (register engine adapters)
from reaxkit.analysis.electrostatics.electrostatics import (
    PolarizationFieldRequest,
    PolarizationFieldTask,
)
from reaxkit.core.platform.engine_resolver import resolve_engine
from reaxkit.core.platform.exceptions import ParseError
from reaxkit.domain.data_models import (
    ChargeData,
    ConnectivityData,
    ElectricFieldData,
    ElectrostaticsData,
    TrajectoryData,
)

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

    task = PolarizationFieldTask()
    task_name = str(task.__class__.__name__).replace("(", "").replace(")", "")
    task_artifacts_dir = ARTIFACTS_DIR / task_name
    task_artifacts_dir.mkdir(parents=True, exist_ok=True)

    load_args = {
        "run_dir": str(run_dir),
        "project_root": str(project_root),
        "xmolout": str(run_dir / "xmolout"),
        "fort7": str(run_dir / "fort.7"),
        "fort78": str(run_dir / "fort.78"),
        "control": str(run_dir / "control"),
        "cache": False,
    }
    try:
        data = ElectrostaticsData(
            trajectory=adapter.load(TrajectoryData, load_args),
            charges=adapter.load(ChargeData, load_args),
            connectivity=adapter.load(ConnectivityData, load_args),
            electric_field=adapter.load(ElectricFieldData, load_args),
        )
    except ParseError as exc:
        if "fort.78" in str(exc):
            pytest.skip("ElectricFieldData source fort.78 is not available for this run_dir.")
        raise

    request = PolarizationFieldRequest(
        frames=None,
        every=1000,
        aggregate="mean",
        field_direction="z",
        dipole_or_polaization_direction="p_z",
    )
    result = task.run(data, request)

    assert result.request == request
    assert isinstance(result.full_table, pd.DataFrame)
    assert isinstance(result.aggregated_table, pd.DataFrame)
    assert {"iter", "mu_z (debye)", "P_z (uC/cm^2)", "field_z"}.issubset(set(result.full_table.columns))
    assert {"field_z", "P_z (uC/cm^2)"}.issubset(set(result.aggregated_table.columns))

    payload = {
        "full_table": result.full_table.head(20).to_dict(orient="records"),
        "aggregated_table": result.aggregated_table.head(20).to_dict(orient="records"),
        "request": asdict(result.request),
    }
    views = task.recommended_presentations(result, payload)
    assert len(views) >= 3
    assert views[0].view_type == "table"
    assert views[1].view_type == "table"
    assert views[2].view_type == "plot2d"

    metadata_path = task_artifacts_dir / "polarization_field_summary.txt"
    full_csv_path = task_artifacts_dir / "polarization_field_full.csv"
    aggregated_csv_path = task_artifacts_dir / "polarization_field_aggregated.csv"
    head_path = task_artifacts_dir / "polarization_field_head.txt"

    metadata_path.write_text(
        "\n".join(
            [
                f"Detected adapter: {adapter.__class__.__name__}",
                f"Result type: {type(result).__name__}",
                f"Full columns: {list(result.full_table.columns)}",
                f"Full rows: {len(result.full_table)}",
                f"Aggregated columns: {list(result.aggregated_table.columns)}",
                f"Aggregated rows: {len(result.aggregated_table)}",
                f"Request aggregate: {result.request.aggregate}",
                f"Request field_direction: {result.request.field_direction}",
                f"Request dipole_or_polaization_direction: {result.request.dipole_or_polaization_direction}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result.full_table.to_csv(full_csv_path, index=False)
    result.aggregated_table.to_csv(aggregated_csv_path, index=False)
    head_path.write_text(
        result.aggregated_table.head(12).to_string(index=False) + "\n",
        encoding="utf-8",
    )
    return task_artifacts_dir


def test_polarization_field_saves_artifacts() -> None:
    if not RUN_DIR.exists():
        pytest.skip(f"RUN_DIR does not exist: {RUN_DIR}")
    out_dir = _run_and_save()
    assert (out_dir / "polarization_field_summary.txt").exists()
    assert (out_dir / "polarization_field_full.csv").exists()
    assert (out_dir / "polarization_field_aggregated.csv").exists()
    assert (out_dir / "polarization_field_head.txt").exists()


def main() -> None:
    if not RUN_DIR.exists():
        return
    _run_and_save()


if __name__ == "__main__":
    main()
