from __future__ import annotations

from pathlib import Path

import numpy as np

from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.core.platform.engine_resolver import resolve_engine
from reaxkit.core.task_registry import TASK_REGISTRY
from reaxkit.domain.data_models import MSDRequest, TrajectoryData
from reaxkit.analysis.trajectory.msd_task import MSDTask


def _write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _minimal_xmolout_two_frames() -> str:
    return (
        "2\n"
        "simA 10 -10.0 1.0 1.0 1.0 90.0 90.0 90.0\n"
        "Ga 0.000 0.000 0.000\n"
        "N 1.000 0.000 0.000\n"
        "2\n"
        "simA 11 -9.5 1.0 1.0 1.0 90.0 90.0 90.0\n"
        "Ga 0.500 0.000 0.000\n"
        "N 1.500 0.000 0.000\n"
    )


def test_task_registry_contains_msd():
    assert "msd" in TASK_REGISTRY


def test_reaxff_adapter_loads_trajectory_data(tmp_path: Path):
    x = _write_text(tmp_path / "xmolout", _minimal_xmolout_two_frames())
    adapter = resolve_engine(str(tmp_path), engine=None)
    data = adapter.load(TrajectoryData, {"xmolout": str(x), "input": str(tmp_path)})

    assert isinstance(data, TrajectoryData)
    assert data.positions.shape == (2, 2, 3)


def test_executor_runs_engine_agnostic_msd_task(tmp_path: Path):
    x = _write_text(tmp_path / "xmolout", _minimal_xmolout_two_frames())

    executor = AnalysisExecutor()
    result = executor.run(
        MSDTask(),
        MSDRequest(atom_ids=[1]),
        {"engine": "reaxff", "input": str(tmp_path), "xmolout": str(x)},
    )

    assert result.table["frame_index"].tolist() == [0, 1]
    assert result.table["atom_id"].tolist() == [1, 1]
    assert np.isclose(result.table["msd"].iloc[0], 0.0)
    assert np.isclose(result.table["msd"].iloc[1], 0.25)
