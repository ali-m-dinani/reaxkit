import argparse
import builtins
import json
import pstats
import threading
import time

import pytest

from reaxkit.core.runtime.cli_policy import add_execution_arguments
from reaxkit.core.runtime.command_diagnostics import CommandDiagnostics
from reaxkit.core.runtime.command_profile import CommandProfile


def measured_leaf():
    time.sleep(0.01)


def measured_recursive(depth):
    if depth:
        measured_recursive(depth - 1)


@pytest.mark.parametrize("clock", ["wall", "cpu"])
def test_real_calls_threads_recursion_and_timings(tmp_path, clock):
    yappi = pytest.importorskip("yappi")
    with CommandDiagnostics(tmp_path, "profile-test", [], "test", profile=True, profile_clock=clock) as report:
        for iteration in range(3):
            measured_leaf()
        worker = threading.Thread(target=measured_leaf)
        worker.start()
        worker.join()
        measured_recursive(3)
        report.exit_code = 0
    data = json.loads((report.directory / "profile.json").read_text())
    leaf = next(row for row in data["functions"] if row["function"] == "measured_leaf")
    recursive = next(row for row in data["functions"] if row["function"] == "measured_recursive")
    assert leaf["calls"] == leaf["primitive_calls"] == 4
    assert recursive["calls"] == 4
    assert recursive["primitive_calls"] == 1
    assert leaf["cumulative_seconds"] >= leaf["self_seconds"] >= 0
    if clock == "wall":
        assert leaf["cumulative_seconds"] >= 0.04
    assert len(data["threads"]) >= 2
    assert data["clock"] == clock
    assert "measured_leaf" in (report.directory / "profile_summary.txt").read_text()
    assert pstats.Stats(str(report.directory / "profile.pstat")).total_calls > 0
    assert not yappi.is_running()
    assert not yappi.get_func_stats()


def test_profile_finalizes_on_exception(tmp_path):
    pytest.importorskip("yappi")
    with pytest.raises(RuntimeError, match="test failure"):
        with CommandDiagnostics(tmp_path, "failure", [], "test", profile=True) as report:
            measured_leaf()
            raise RuntimeError("test failure")
    assert (report.directory / "profile.pstat").exists()


def test_profile_requires_diagnostics(tmp_path):
    with pytest.raises(ValueError, match="requires diagnostics"):
        CommandDiagnostics(tmp_path, "test", [], "test", profile=True, enabled=False)


def test_missing_profiler_message(monkeypatch):
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "yappi":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(ValueError, match=r"reaxkit\[profiling\]"):
        CommandProfile()


def test_existing_session_not_destroyed():
    yappi = pytest.importorskip("yappi")
    yappi.start()
    try:
        with pytest.raises(ValueError, match="Another profiler"):
            CommandProfile()
        assert yappi.is_running()
    finally:
        yappi.stop()
        yappi.clear_stats()


def test_flags_before_and_after_command():
    parser = argparse.ArgumentParser()
    parser.add_subparsers(dest="command").add_parser("test")
    add_execution_arguments(parser)
    for argv in (["--profile", "--profile-clock", "cpu", "test"],
                 ["test", "--profile", "--profile-clock", "cpu"]):
        args = parser.parse_args(argv)
        assert args.profile is True
        assert args.profile_clock == "cpu"
    assert parser.parse_args(["test"]).profile is False
