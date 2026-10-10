"""Sanity check for SimulationScalarSeriesTask via AnalysisExecutor."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import reaxkit.engine  # noqa: F401 (register engine adapters)
from reaxkit.analysis.timeseries.timeseries import SimulationScalarSeriesRequest, SimulationScalarSeriesTask
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.core.platform.engine_resolver import resolve_engine
from reaxkit.domain.data_models import SimulationData

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

    task = SimulationScalarSeriesTask()
    task_name = str(task.__class__.__name__).replace("(", "").replace(")", "")
    task_artifacts_dir = ARTIFACTS_DIR / task_name
    task_artifacts_dir.mkdir(parents=True, exist_ok=True)
    request = SimulationScalarSeriesRequest(
        field="potential_energy",
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
    assert {"frame_index", "iter", "field", "value"}.issubset(set(result.table.columns))

    metadata_path = task_artifacts_dir / "simulation_scalar_series_summary.txt"
    csv_path = task_artifacts_dir / "simulation_scalar_series.csv"
    head_path = task_artifacts_dir / "simulation_scalar_series_head.txt"

    metadata_path.write_text(
        "\n".join(
            [
                f"Detected adapter: {adapter.__class__.__name__}",
                f"Result type: {type(result).__name__}",
                f"Columns: {list(result.table.columns)}",
                f"Rows: {len(result.table)}",
                f"Request field: {result.request.field}",
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


def test_simulation_scalar_series_saves_artifacts() -> None:
    if not RUN_DIR.exists():
        pytest.skip(f"RUN_DIR does not exist: {RUN_DIR}")
    out_dir = _run_and_save()
    assert (out_dir / "simulation_scalar_series_summary.txt").exists()
    assert (out_dir / "simulation_scalar_series.csv").exists()
    assert (out_dir / "simulation_scalar_series_head.txt").exists()


def test_potential_energy_per_atom_uses_each_frames_atom_count() -> None:
    data = SimulationData(
        atom_ids=[1, 2, 3, 4],
        iterations=np.asarray([0, 10, 20]),
        potential_energy=np.asarray([-40.0, -30.0, -10.0]),
        num_of_atoms=np.asarray([4, 3, 2]),
    )
    request = SimulationScalarSeriesRequest(
        field="potential_energy",
        per_atom=True,
    )

    result = SimulationScalarSeriesTask().run(data, request)

    assert result.table["field"].tolist() == ["potential_energy_per_atom"] * 3
    assert result.table["value"].tolist() == [-10.0, -10.0, -5.0]


def test_potential_energy_per_atom_requests_atom_counts_from_loader() -> None:
    request = SimulationScalarSeriesRequest(
        field="potential_energy",
        per_atom=True,
    )

    assert SimulationScalarSeriesTask.required_data_fields_for(request, {}) == (
        "potential_energy",
        "num_of_atoms",
    )


@pytest.mark.parametrize("field", ["elapsed_time", "elap_time"])
@pytest.mark.parametrize("frames,every", [(None, 1), (None, 2), ([3, 1], 1)])
def test_elapsed_time_per_iter_divides_by_iteration(field, frames, every) -> None:
    data = SimulationData(
        atom_ids=[],
        iterations=np.asarray([0, 10, 30, 40]),
        elapsed_time=np.asarray([5.0, 25.0, 85.0, 125.0]),
    )
    result = SimulationScalarSeriesTask().run(
        data, SimulationScalarSeriesRequest(field=field, frames=frames, every=every)
    )
    selected = result.table["frame_index"].to_numpy()
    np.testing.assert_allclose(result.table["value"], data.elapsed_time[selected])
    np.testing.assert_allclose(
        result.table["elapsed_time_per_iter"],
        np.asarray([np.nan, 2.5, 85.0 / 30, 125.0 / 40])[selected],
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "iterations,elapsed,expected",
    [
        ([0], [5.0], [np.nan]),
        ([10], [5.0], [0.5]),
        (None, [5.0, 10.0], [np.nan, np.nan]),
        ([0, 0, 10, 5, 15, 25], [5, 10, 20, 25, 2, 12],
         [np.nan, np.nan, 2.0, 5.0, 2.0 / 15, 12.0 / 25]),
        ([0, 10, 20, 30], [5, np.nan, 15, 15], [np.nan, np.nan, 0.75, 0.5]),
    ],
)
def test_elapsed_time_per_iter_handles_zero_and_missing_values(iterations, elapsed, expected) -> None:
    data = SimulationData(
        atom_ids=[],
        iterations=None if iterations is None else np.asarray(iterations),
        elapsed_time=np.asarray(elapsed),
    )
    result = SimulationScalarSeriesTask().run(
        data, SimulationScalarSeriesRequest(field="elapsed_time")
    )
    np.testing.assert_allclose(
        result.table["elapsed_time_per_iter"], expected, equal_nan=True
    )


@pytest.mark.parametrize("suffix", ["", "s"])
def test_elapsed_time_per_iter_from_summary_without_trajectory(tmp_path, monkeypatch, suffix) -> None:
    summary = tmp_path / "summary.txt"
    summary.write_text(
        f"0 1 0 -10 100 300 1 1 5{suffix}\n"
        f"10 1 1 -10 100 300 1 1 25{suffix}\n"
        f"30 1 3 -10 100 300 1 1 85{suffix}\n",
        encoding="utf-8",
    )
    result = AnalysisExecutor().run(
        SimulationScalarSeriesTask(),
        SimulationScalarSeriesRequest(field="elapsed_time", every=2),
        {
            "engine": "reaxff",
            "summary": str(summary),
            "run_dir": str(tmp_path),
            "project_root": str(tmp_path / "workspace"),
            "cache": False,
        },
    )
    np.testing.assert_allclose(
        result.table["elapsed_time_per_iter"], [np.nan, 85.0 / 30], equal_nan=True
    )
    from reaxkit.workflows.timeseries.get_elapsed_time import build_parser, run_main

    monkeypatch.chdir(tmp_path)
    parser = build_parser(argparse.ArgumentParser(), command="get_elapsed_time")
    args = parser.parse_args([
        "--engine", "reaxff", "--summary", "summary.txt", "--plot", "single",
        "--export", str(tmp_path / "elapsed.csv"),
        "--save", str(tmp_path / "elapsed.png"),
    ])
    args.project_root = str(tmp_path / "workspace")
    args.cache = False
    assert run_main("get_elapsed_time", args) == 0
    exported = pd.read_csv(tmp_path / "elapsed.csv")
    np.testing.assert_allclose(exported["value"], [5.0, 25.0, 85.0])
    np.testing.assert_allclose(
        exported["elapsed_time_per_iter"], [np.nan, 2.5, 85.0 / 30], equal_nan=True
    )
    assert (tmp_path / "elapsed.png").is_file()


@pytest.mark.parametrize("mode", ["single", "subplot", "separate"])
def test_elapsed_time_plot_includes_both_series(mode, tmp_path) -> None:
    from reaxkit.workflows.timeseries.common import build_plot_payload
    from reaxkit.presentation.plot.registry import plot

    result = SimulationScalarSeriesTask().run(
        SimulationData(
            atom_ids=[], iterations=np.asarray([0, 10, 30]),
            elapsed_time=np.asarray([0.0, 25.0, 85.0]),
        ),
        SimulationScalarSeriesRequest(field="elapsed_time"),
    )
    payload = build_plot_payload(
        "get_elapsed_time", result, argparse.Namespace(plot=mode, xaxis="iter")
    )
    if mode == "separate":
        series = [item["series"][0] for item in payload]
        assert [item["ylabel"] for item in payload] == [
            "Elapsed time (s)", "Elapsed time per iteration (s/iter)"
        ]
    elif mode == "subplot":
        series = [panel[0] for panel in payload["subplots"]]
        assert payload["ylabel"] == [
            "Elapsed time (s)", "Elapsed time per iteration (s/iter)"
        ]
    else:
        series = payload["series"]
        assert payload["legend"]
        assert payload["ylabel"] == "Elapsed time (s) / elapsed time per iteration (s/iter)"
    assert [item["label"] for item in series] == ["elapsed_time", "elapsed_time_per_iter"]
    np.testing.assert_allclose(series[0]["y"], [0.0, 25.0, 85.0])
    np.testing.assert_allclose(series[1]["y"], [np.nan, 2.5, 85.0 / 30], equal_nan=True)
    assert all(item["x"] == [0, 10, 30] for item in series)
    payloads = payload if mode == "separate" else [payload]
    rendered_lines = []
    for index, item in enumerate(payloads):
        figure = plot({**item, "save": str(tmp_path / f"plot_{index}.png")})
        rendered_lines.extend(line for axis in figure.axes for line in axis.lines)
    assert len(rendered_lines) == 2
    for line, expected in zip(rendered_lines, series):
        np.testing.assert_allclose(line.get_ydata(), expected["y"], equal_nan=True)
        assert np.isfinite(line.get_ydata()).any()


def main() -> None:
    if not RUN_DIR.exists():
        return
    _run_and_save()


if __name__ == "__main__":
    main()
