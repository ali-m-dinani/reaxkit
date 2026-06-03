from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.engine.reaxff.adapter import ReaxFFAdapter


class _DummyTask:
    required_data = object

    def run(self, data, request):
        return data


def test_analysis_executor_prefers_xmolout_for_detection(monkeypatch):
    captured = {}

    class DummyAdapter:
        def load(self, required_data, args, reporter=None):
            return {"ok": True}

    def fake_resolve_engine(path, engine=None):
        captured["path"] = path
        captured["engine"] = engine
        return DummyAdapter()

    monkeypatch.setattr("reaxkit.core.analysis_executor.resolve_engine", fake_resolve_engine)

    executor = AnalysisExecutor()
    executor.run(
        _DummyTask(),
        request=object(),
        args={"xmolout": "renamed_xmolout", "run_dir": "../sim1", "no_cache": True},
    )

    assert captured["path"] == "renamed_xmolout"
    assert captured["engine"] is None


def test_analysis_executor_uses_run_dir_when_no_file_hint(monkeypatch):
    captured = {}

    class DummyAdapter:
        def load(self, required_data, args, reporter=None):
            return {"ok": True}

    def fake_resolve_engine(path, engine=None):
        captured["path"] = path
        return DummyAdapter()

    monkeypatch.setattr("reaxkit.core.analysis_executor.resolve_engine", fake_resolve_engine)

    executor = AnalysisExecutor()
    executor.run(_DummyTask(), request=object(), args={"run_dir": "../sim1", "no_cache": True})

    assert captured["path"] == "../sim1"


def test_reaxff_adapter_load_trajectory_uses_run_dir_default(monkeypatch, tmp_path):
    sim_dir = tmp_path / "sim1"
    sim_dir.mkdir()

    captured = {}

    class DummyHandler:
        def __init__(self, path, reporter=None):
            captured["path"] = Path(path)

        def n_frames(self):
            return 1

        def frame(self, index):
            return {"coords": np.zeros((1, 3)), "atom_types": ["H"]}

        def dataframe(self):
            return pd.DataFrame({"iter": [0], "a": [1.0], "b": [1.0], "c": [1.0]})

    monkeypatch.setattr("reaxkit.engine.reaxff.adapter.XmoloutHandler", DummyHandler)
    monkeypatch.setattr(ReaxFFAdapter, "_load_simulation_from_summary", staticmethod(lambda args, reporter=None: None))

    adapter = ReaxFFAdapter()
    adapter.load_trajectory({"run_dir": str(sim_dir)})

    assert captured["path"] == sim_dir / "xmolout"


def test_reaxff_detect_accepts_renamed_xmolout_file(tmp_path):
    renamed = tmp_path / "renamed_xmolout"
    renamed.write_text("dummy", encoding="utf-8")

    adapter = ReaxFFAdapter()

    assert adapter.detect(renamed) > 0.0


def test_analysis_executor_uses_task_required_data_for(monkeypatch):
    captured = {}

    class DummyAdapter:
        def required_input_files(self, data_type, args):
            captured["required_input_files_data_type"] = data_type
            return ()

        def load(self, data_type, args, reporter=None):
            captured["load_data_type"] = data_type
            return "payload"

    class DynamicTask:
        required_data = object

        @staticmethod
        def required_data_for(request, args):
            _ = (request, args)
            return str

        def run(self, data, request):
            _ = request
            return data

    def fake_resolve_engine(path, engine=None):
        _ = (path, engine)
        return DummyAdapter()

    monkeypatch.setattr("reaxkit.core.analysis_executor.resolve_engine", fake_resolve_engine)

    executor = AnalysisExecutor()
    result = executor.run(
        DynamicTask(),
        request=object(),
        args={"xmolout": "renamed_xmolout", "no_cache": True},
    )

    assert result == "payload"
    assert captured["required_input_files_data_type"] is str
    assert captured["load_data_type"] is str
