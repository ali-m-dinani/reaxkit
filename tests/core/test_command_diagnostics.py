import argparse
import json
import subprocess
import sys
import time

import pytest

from reaxkit.core.runtime.cli_policy import add_execution_arguments
from reaxkit.core.runtime.command_diagnostics import CommandDiagnostics


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


def test_native_gil_stall_is_sampled(tmp_path):
    code = '''
import ctypes, os, sys
from reaxkit.core.runtime.command_diagnostics import CommandDiagnostics
with CommandDiagnostics(sys.argv[1], "native-stall", [], "test", interval=0.2) as report:
    if os.name == "nt":
        ctypes.PyDLL("kernel32").Sleep(1200)
    else:
        ctypes.PyDLL(None).sleep(1)
    report.exit_code = 0
'''
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    directory = next((tmp_path / "logs" / "diagnostics").iterdir())
    stacks = (directory / "stacks.log").read_text()
    assert stacks.count("=== Stack sample") >= 2
    assert 'File "<string>"' in stacks
    assert "elapsed" in stacks


def test_abrupt_exit_keeps_native_traces(tmp_path):
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
    for argv in (["--no-diagnostics", "test"], ["test", "--no-diagnostics"]):
        assert parser.parse_args(argv).diagnostics is False


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("profile", [False, True])
def test_cli_report_contains_execution_steps_and_handled_errors(tmp_path, monkeypatch, fail, profile):
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
    monkeypatch.setattr(cli, "build_parser", lambda command: parser)
    monkeypatch.setattr(sys, "argv", ["reaxkit"])
    assert cli.main(announce=False) == (4 if fail else 0)
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
