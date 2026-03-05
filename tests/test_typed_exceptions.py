from __future__ import annotations

from pathlib import Path

import pytest

from reaxkit.core.analysis_executor import AnalysisExecutor
from reaxkit.core.exceptions import AnalysisError, ParseError
from reaxkit.engine.reaxff.io.base import BaseHandler


class _BrokenHandler(BaseHandler):
    def _parse(self):
        raise ValueError("bad format")


def test_basehandler_wraps_generic_parse_errors(tmp_path: Path):
    src = tmp_path / "dummy.txt"
    src.write_text("x", encoding="utf-8")
    h = _BrokenHandler(src)
    with pytest.raises(ParseError, match="Failed to parse"):
        h.parse()


class _DummyTask:
    required_data = dict

    def run(self, data, request):
        _ = (data, request)
        return {"ok": True}


def test_analysis_executor_wraps_load_errors_as_parseerror(monkeypatch, tmp_path: Path):
    class BrokenAdapter:
        def load(self, required_data, args, reporter=None):
            _ = (required_data, args, reporter)
            raise ValueError("loader failed")

    def fake_resolve_engine(path, engine=None):
        _ = (path, engine)
        return BrokenAdapter()

    monkeypatch.setattr("reaxkit.core.analysis_executor.resolve_engine", fake_resolve_engine)
    args = {"project_root": str(tmp_path), "input": str(tmp_path), "run_dir": str(tmp_path), "no_cache": True}
    with pytest.raises(ParseError, match="Failed to load required data"):
        AnalysisExecutor().run(_DummyTask(), request=object(), args=args)


def test_analysis_executor_wraps_task_errors_as_analysiserror(monkeypatch, tmp_path: Path):
    class OkAdapter:
        def load(self, required_data, args, reporter=None):
            _ = (required_data, args, reporter)
            return {"ok": True}

    class BrokenTask(_DummyTask):
        def run(self, data, request):
            _ = (data, request)
            raise RuntimeError("task failed")

    def fake_resolve_engine(path, engine=None):
        _ = (path, engine)
        return OkAdapter()

    monkeypatch.setattr("reaxkit.core.analysis_executor.resolve_engine", fake_resolve_engine)
    args = {"project_root": str(tmp_path), "input": str(tmp_path), "run_dir": str(tmp_path), "no_cache": True}
    with pytest.raises(AnalysisError, match="failed during analysis"):
        AnalysisExecutor().run(BrokenTask(), request=object(), args=args)
