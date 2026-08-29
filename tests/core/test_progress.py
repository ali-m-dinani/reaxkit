from __future__ import annotations

import numpy as np

from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.core.runtime import progress
from reaxkit.domain.data_models import TrajectoryData
from reaxkit.engine.base import EngineAdapter


class _QuietAdapter(EngineAdapter):
    def detect(self, path):
        _ = path
        return 1.0

    def load_trajectory(self, args, reporter=None):
        _ = (args, reporter)
        return TrajectoryData(
            positions=np.zeros((1, 1, 3), dtype=float),
            elements=["H"],
            atom_ids=[1],
        )


class _QuietTask:
    def run(self, data, request, reporter=None):
        _ = (data, request, reporter)
        return "done"


def test_progress_flag_wins_over_quiet_log_level(monkeypatch):
    def sentinel(stage, current, total, message=None):
        _ = (stage, current, total, message)

    monkeypatch.setattr(progress, "tqdm_reporter_factory", lambda: sentinel)

    reporter = progress.resolve_reporter({"progress": True, "log": "quiet"})

    assert reporter is sentinel


def test_explicit_quiet_still_disables_progress():
    reporter = progress.resolve_reporter({"progress": True, "quiet": True})

    assert reporter is progress.noop_reporter


def test_progress_operation_supplies_fallback_lifecycle_events():
    events: list[tuple[str, int, int, str | None]] = []

    def reporter(stage, current, total, message=None):
        events.append((stage, current, total, message))

    with progress.progress_operation(reporter, "load", "Loading data", "Loaded data"):
        pass

    assert events == [
        ("load", 0, 0, "Loading data"),
        ("load", 1, 1, "Loaded data"),
    ]


def test_progress_operation_forwards_determinate_events_without_duplicate_finish():
    events: list[tuple[str, int, int, str | None]] = []

    def reporter(stage, current, total, message=None):
        events.append((stage, current, total, message))

    with progress.progress_operation(reporter, "analyze", "Starting", "Finished") as operation:
        operation("analyze", 1, 2, "Frame 1")
        operation("analyze", 2, 2, "Frame 2")

    assert events == [
        ("analyze", 0, 0, "Starting"),
        ("analyze", 1, 2, "Frame 1"),
        ("analyze", 2, 2, "Frame 2"),
    ]


def test_progress_operation_closes_indeterminate_stream_at_processed_count():
    events: list[tuple[str, int, int, str | None]] = []

    def reporter(stage, current, total, message=None):
        events.append((stage, current, total, message))

    with progress.progress_operation(reporter, "stream", "Starting", "Finished") as operation:
        operation("stream", 1, 0, "Frame 1")
        operation("stream", 7, 0, "Frame 7")

    assert events[-1] == ("stream", 7, 7, "Finished")


def test_engine_adapter_guarantees_load_progress_for_quiet_loaders():
    events: list[tuple[str, int, int, str | None]] = []

    def reporter(stage, current, total, message=None):
        events.append((stage, current, total, message))

    data = _QuietAdapter().load(TrajectoryData, {}, reporter=reporter)

    assert data.positions.shape == (1, 1, 3)
    assert events[0][:3] == ("load", 0, 0)
    assert events[-1][:3] == ("load", 1, 1)


def test_analysis_executor_guarantees_analysis_progress_for_quiet_tasks():
    events: list[tuple[str, int, int, str | None]] = []

    def reporter(stage, current, total, message=None):
        events.append((stage, current, total, message))

    result = AnalysisExecutor._run_task(_QuietTask(), object(), object(), reporter)

    assert result == "done"
    assert events[0][:3] == ("analyze", 0, 0)
    assert events[-1][:3] == ("analyze", 1, 1)


def test_tqdm_reporter_suppresses_duplicate_terminal_callback(monkeypatch):
    created = []

    class FakeBar:
        def __init__(self, total=None, **kwargs):
            _ = kwargs
            self.total = total
            self.n = 0
            created.append(self)

        def set_description_str(self, desc):
            self.desc = desc

        def refresh(self):
            pass

        def update(self, delta):
            self.n += delta

        def reset(self, total=None):
            self.total = total
            self.n = 0

        def close(self):
            pass

    monkeypatch.setattr(progress, "tqdm", FakeBar)
    reporter = progress.tqdm_reporter_factory()

    reporter("load", 0, 0, "Loading")
    reporter("load", 3, 3, "Parsing frames")
    reporter("load", 3, 3, "Finished parsing frames")

    assert len(created) == 1
