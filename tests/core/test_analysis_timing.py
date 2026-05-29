from __future__ import annotations

import json
from pathlib import Path

from reaxkit.core.runtime.analysis_executor import AnalysisExecutor


class _DummyTask:
    required_data = dict

    def run(self, data, request):
        _ = request
        return {"ok": bool(data.get("ok"))}


def test_analysis_executor_writes_timing_log_by_default(monkeypatch, tmp_path: Path):
    captured = {}

    class DummyAdapter:
        def load(self, required_data, args, reporter=None):
            _ = (required_data, reporter)
            captured["args"] = dict(args)
            return {"ok": True}

    def fake_resolve_engine(path, engine=None):
        _ = (path, engine)
        return DummyAdapter()

    monkeypatch.setattr("reaxkit.core.analysis_executor.resolve_engine", fake_resolve_engine)

    args = {
        "project_root": str(tmp_path),
        "input": str(tmp_path),
        "run_dir": str(tmp_path),
        "no_cache": True,
    }
    out = AnalysisExecutor().run(_DummyTask(), request=object(), args=args)
    assert out == {"ok": True}

    timing_log = Path(args["project_root"]) / "logs" / "timing.log"
    assert timing_log.exists()

    lines = [ln for ln in timing_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 2
    records = [json.loads(ln) for ln in lines]
    phases = {rec.get("phase") for rec in records}
    assert "load" in phases
    assert "analyze" in phases

    logs_dir = Path(args["project_root"]) / "logs"
    assert (logs_dir / "reaxkit.log").exists()
    run_logs = list(logs_dir.glob("run_*.log"))
    assert run_logs
    assert (logs_dir / "timing_human.log").exists()
    run_timing_logs = list(logs_dir.glob("run_*.timing.log"))
    assert run_timing_logs
