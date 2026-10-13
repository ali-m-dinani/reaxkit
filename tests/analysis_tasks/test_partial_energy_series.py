"""Sanity check for PartialEnergySeriesTask via AnalysisExecutor."""

from __future__ import annotations

from pathlib import Path

import pytest
import numpy as np
from reaxkit.domain.data_models import PartialEnergyData
import reaxkit.engine  # noqa: F401 (register engine adapters)
from reaxkit.analysis.timeseries.timeseries import PartialEnergySeriesRequest, PartialEnergySeriesTask
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.core.platform.exceptions import AnalysisError
from reaxkit.core.platform.engine_resolver import resolve_engine

RUN_DIR = Path(
    r"C:\Users\alimo\PycharmProjects\pythonProject\reaxkit\examples_to_test"
)
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


@pytest.mark.parametrize("num_atoms", [None, 10])
@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("reference_frame", [0, 1])
def test_partial_energy_normalization_and_reference(num_atoms, relative, reference_frame):
    data = PartialEnergyData(
        iterations=np.asarray([0, 10, 20]), components=("Ebond", "Eatom"),
        values=np.asarray([[-40., 20.], [-30., 40.], [-10., 80.]]),
    )
    request = PartialEnergySeriesRequest(
        num_atoms=num_atoms, relative=relative, reference_frame=reference_frame,
        frames=[2, 1], every=2,
    )
    table = PartialEnergySeriesTask().run(data, request).table.set_index("component")
    divisor = num_atoms or 1
    assert table["frame_index"].tolist() == [2, 2]
    assert table.loc["Ebond", "value"] == -10 / divisor
    assert table.loc["Eatom", "value"] == 80 / divisor
    if relative:
        assert table.loc["Ebond", "relative_energy"] == (30 if reference_frame == 0 else 20) / divisor
        assert table.loc["Eatom", "relative_energy"] == (60 if reference_frame == 0 else 40) / divisor
    else:
        assert "relative_energy" not in table


@pytest.mark.parametrize("options,match", [
    ({"num_atoms": 0}, "positive integer"),
    ({"num_atoms": -1}, "positive integer"),
    ({"per_atom": True}, "requires --num-atoms"),
    ({"relative": True, "reference_frame": -1}, "Reference frame"),
    ({"relative": True, "reference_frame": 1}, "Reference frame"),
])
def test_partial_energy_invalid_normalization(options, match):
    data = PartialEnergyData(iterations=np.asarray([0]), components=("Ebond",), values=np.asarray([[-40.]]))
    with pytest.raises(ValueError, match=match):
        PartialEnergySeriesTask().run(data, PartialEnergySeriesRequest(**options))


def _run_and_save() -> Path:
    run_dir = RUN_DIR
    if not run_dir.exists():
        raise FileNotFoundError(f"RUN_DIR does not exist: {run_dir}")
    project_root = run_dir / "reaxkit_workspace"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    adapter = resolve_engine(str(run_dir), engine=None)

    task = PartialEnergySeriesTask()
    task_name = str(task.__class__.__name__).replace("(", "").replace(")", "")
    task_artifacts_dir = ARTIFACTS_DIR / task_name
    task_artifacts_dir.mkdir(parents=True, exist_ok=True)
    request = PartialEnergySeriesRequest(
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
        if "expected data type PartialEnergyData, got NoneType" in msg or "Failed to load required data 'PartialEnergyData'" in msg:
            pytest.skip("PartialEnergyData is not available for this run_dir.")
        raise

    assert result.request == request
    assert {"frame_index", "iter", "component", "value"}.issubset(set(result.table.columns))

    metadata_path = task_artifacts_dir / "partial_energy_series_summary.txt"
    csv_path = task_artifacts_dir / "partial_energy_series.csv"
    head_path = task_artifacts_dir / "partial_energy_series_head.txt"

    metadata_path.write_text(
        "\n".join(
            [
                f"Detected adapter: {adapter.__class__.__name__}",
                f"Result type: {type(result).__name__}",
                f"Columns: {list(result.table.columns)}",
                f"Rows: {len(result.table)}",
                f"Request components: {list(result.request.components) if result.request.components is not None else None}",
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


def test_partial_energy_series_saves_artifacts() -> None:
    if not RUN_DIR.exists():
        pytest.skip(f"RUN_DIR does not exist: {RUN_DIR}")
    out_dir = _run_and_save()
    assert (out_dir / "partial_energy_series_summary.txt").exists()
    assert (out_dir / "partial_energy_series.csv").exists()
    assert (out_dir / "partial_energy_series_head.txt").exists()


def main() -> None:
    if not RUN_DIR.exists():
        return
    _run_and_save()


if __name__ == "__main__":
    main()
