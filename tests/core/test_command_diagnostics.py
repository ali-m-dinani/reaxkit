import argparse
import faulthandler
import json
import subprocess
import sys
import threading
import time

import pytest

from reaxkit.core.runtime.cli_policy import add_execution_arguments
from reaxkit.core.runtime.command_diagnostics import CommandDiagnostics, _StackSampler


def test_console_metadata_and_cleanup(tmp_path, capsys):
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with CommandDiagnostics(tmp_path, "test-command", ["test-command"], "test", interval=0.1) as report:
        print("ordinary output")
        print("error output", file=sys.stderr)
        time.sleep(0.35)
        report.exit_code = 0
    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    console = (report.directory / "console.log").read_text()
    assert "ordinary output" in console and "error output" in console
    metadata = json.loads((report.directory / "report.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["elapsed_seconds"] >= 0.35
    assert metadata["argv"] == ["test-command"]
    assert "elapsed" in (report.directory / "stacks.log").read_text()
    assert "test_console_metadata_and_cleanup" in (report.directory / "stacks.log").read_text()
    assert report.observer.poll() == 0
    assert not report.sampler.is_alive()
    assert report.raw.closed
    assert metadata["stack_sampling"]["requires_gil"] is True
    assert "error" not in metadata["stack_sampling"]
    assert f"Thread {report.sampler.ident:#x}" not in (report.directory / "stacks.raw.log").read_text()
    summary = json.loads((report.directory / "stack_summary.json").read_text())
    assert summary["samples"] >= 1
    assert any(row["thread"] == "main" for row in summary["functions"])
    assert (report.directory / "resources.jsonl").read_text().strip()


def test_failure_and_disabled_reports(tmp_path):
    with pytest.raises(ValueError, match="scientific failure"):
        with CommandDiagnostics(tmp_path, "test", [], "test") as report:
            raise ValueError("scientific failure")
    assert json.loads((report.directory / "report.json").read_text())["status"] == "failed"
    assert "scientific failure" in (report.directory / "exception.log").read_text()
    with CommandDiagnostics(tmp_path, "disabled", [], "test", enabled=False) as disabled:
        pass
    assert not disabled.directory.exists()


def test_native_call_releasing_gil_is_sampled(tmp_path):
    code = '''
import ctypes, os, sys
from reaxkit.core.runtime.command_diagnostics import CommandDiagnostics
with CommandDiagnostics(sys.argv[1], "native-stall", [], "test", interval=0.2) as report:
    if os.name == "nt":
        ctypes.CDLL("kernel32").Sleep(1200)
    else:
        ctypes.CDLL(None).sleep(1)
    report.exit_code = 0
'''
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    directory = next((tmp_path / "logs" / "diagnostics").iterdir())
    stacks = (directory / "stacks.log").read_text()
    assert stacks.count("=== Stack sample") >= 2
    assert 'File "<string>"' in stacks
    assert "elapsed" in stacks


def test_abrupt_exit_keeps_sampled_traces(tmp_path):
    code = '''
import os, sys, time
from reaxkit.core.runtime.command_diagnostics import CommandDiagnostics
with CommandDiagnostics(sys.argv[1], "abrupt", [], "test", interval=0.1):
    time.sleep(0.6)
    os._exit(7)
'''
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 7
    directory = next((tmp_path / "logs" / "diagnostics").iterdir())
    assert (directory / "stacks.raw.log").read_bytes()
    assert json.loads((directory / "report.json").read_text())["status"] == "running"


def test_diagnostics_never_modify_faulthandler(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Diagnostics must not use process-global native faulthandler hooks")

    enabled = faulthandler.is_enabled()
    for name in ("dump_traceback_later", "cancel_dump_traceback_later", "dump_traceback", "enable", "disable"):
        monkeypatch.setattr(faulthandler, name, forbidden)
    with CommandDiagnostics(tmp_path, "safe-sampling", [], "test", interval=0.01) as report:
        time.sleep(0.1)
        report.exit_code = 0
    assert report.sampler.samples > 0
    assert faulthandler.is_enabled() == enabled


def test_resources_continue_during_gil_stall(tmp_path):
    code = '''
import ctypes, json, os, sys, time
from reaxkit.core.runtime.command_diagnostics import CommandDiagnostics
with CommandDiagnostics(sys.argv[1], "gil-stall", [], "test", interval=0.1) as report:
    time.sleep(0.5)
    before = time.monotonic() - report.started
    if os.name == "nt":
        ctypes.PyDLL("kernel32").Sleep(6000)
    else:
        ctypes.PyDLL(None).sleep(6)
    after = time.monotonic() - report.started
    time.sleep(0.3)
    (report.directory / "stall.json").write_text(json.dumps([before, after]))
    report.exit_code = 0
'''
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    directory = next((tmp_path / "logs" / "diagnostics").iterdir())
    before, after = json.loads((directory / "stall.json").read_text())
    resources = [json.loads(line) for line in (directory / "resources.jsonl").read_text().splitlines()]
    assert any(before < row["elapsed_seconds"] < after for row in resources)
    metadata = json.loads((directory / "report.json").read_text())
    assert metadata["status"] == "completed"
    assert "delays samples" in metadata["stack_sampling"]["limitations"]
    assert "missed intervals" in (directory / "stack_summary.txt").read_text()


def test_thread_and_code_churn_in_subprocess(tmp_path):
    code = '''
import sys, threading, time, types
from reaxkit.core.runtime.command_diagnostics import CommandDiagnostics
def template():
    return 1
def worker():
    for iteration in range(100):
        function = types.FunctionType(template.__code__.replace(), {})
        function()
        time.sleep(0)
with CommandDiagnostics(sys.argv[1], "thread-churn", [], "test", interval=0.001) as report:
    for iteration in range(30):
        workers = [threading.Thread(target=worker) for index in range(8)]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join()
    report.exit_code = 0
assert report.sampler.samples > 0
assert not report.sampler.is_alive()
assert report.sampler.error is None
'''
    result = subprocess.run([sys.executable, "-X", "faulthandler", "-c", code, str(tmp_path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    directory = next((tmp_path / "logs" / "diagnostics").iterdir())
    assert json.loads((directory / "report.json").read_text())["status"] == "completed"
    assert "worker" in (directory / "stacks.raw.log").read_text()


def test_sampler_failure_does_not_fail_analysis(tmp_path, monkeypatch):
    attempted = threading.Event()

    def broken_snapshot(self):
        attempted.set()
        raise RuntimeError("sampling unavailable")

    monkeypatch.setattr(_StackSampler, "_snapshot", broken_snapshot)
    with CommandDiagnostics(tmp_path, "sampler-error", [], "test", interval=0.01) as report:
        assert attempted.wait(timeout=5)
        report.exit_code = 0
    metadata = json.loads((report.directory / "report.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["stack_sampling"]["error"] == "RuntimeError: sampling unavailable"
    assert report.raw.closed


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "bad"])
def test_invalid_interval_rejected(value):
    parser = argparse.ArgumentParser()
    add_execution_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--diagnostic-interval-seconds", value])


def test_diagnostic_flags_before_and_after_command():
    parser = argparse.ArgumentParser()
    parser.add_subparsers(dest="command").add_parser("test")
    add_execution_arguments(parser)
    assert parser.parse_args(["test"]).diagnostics is False
    for argv in (["--diagnostics", "test"], ["test", "--diagnostics"]):
        assert parser.parse_args(argv).diagnostics is True
    for argv in (["--no-diagnostics", "test"], ["test", "--no-diagnostics"]):
        assert parser.parse_args(argv).diagnostics is False


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("profile", [False, True])
def test_cli_report_contains_execution_steps_and_handled_errors(tmp_path, monkeypatch, fail, profile, enabled):
    if profile:
        pytest.importorskip("yappi")
    from importlib import import_module
    cli = import_module("reaxkit.cli.main")

    def run(args):
        print("command reached")
        if fail:
            raise FileNotFoundError("missing test input")
        return 0

    parser = argparse.ArgumentParser()
    parser.set_defaults(command="test-command", project_root=str(tmp_path), _run=run, profile=profile)
    if enabled:
        parser.set_defaults(diagnostics=True)
    monkeypatch.setattr(cli, "build_parser", lambda command: parser)
    monkeypatch.setattr(sys, "argv", ["reaxkit"])
    if profile and not enabled:
        with pytest.raises(SystemExit) as error:
            cli.main(announce=False)
        assert error.value.code == 2
        assert not (tmp_path / "logs" / "diagnostics").exists()
        return
    assert cli.main(announce=False) == (4 if fail else 0)
    if not enabled:
        assert not (tmp_path / "logs" / "diagnostics").exists()
        return
    directory = next((tmp_path / "logs" / "diagnostics").iterdir())
    report = json.loads((directory / "report.json").read_text())
    assert report["exit_code"] == (4 if fail else 0)
    assert report["status"] == ("failed" if fail else "completed")
    assert report["execution"]["steps"]
    assert (directory / "steps.log").is_file()
    if profile:
        assert (directory / "profile_summary.txt").is_file()
    if fail:
        assert "FileNotFoundError" in (directory / "exception.log").read_text()
