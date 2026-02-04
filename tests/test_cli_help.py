"""
Tests for CLI help behavior (reaxkit.cli).

These tests validate that:
- top-level -h/--help works
- kind-level help works (e.g., `reaxkit fort7 -h`)
- kind-level workflows that do not require a task (`help`, `intspec`) accept no task
  and wire a default runner
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import reaxkit.cli as cli


def test_top_level_help_prints_usage(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(sys, "argv", ["reaxkit", "-h"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 0

    out = capsys.readouterr().out
    assert "reaxkit CLI" in out
    assert "positional arguments" in out or "options" in out


def test_kind_help_prints_tasks(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(sys, "argv", ["reaxkit", "fort7", "-h"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 0

    out = capsys.readouterr().out
    # Should show kind subcommands (task names) for fort7 workflow
    assert "fort7" in out
    assert "get" in out  # at least one expected task


def test_help_kind_runs_without_task(monkeypatch: pytest.MonkeyPatch):
    # Stub help_workflow.run_main so we don't depend on YAML indices etc.
    called = {"ok": False}

    def _stub_run_main(args):
        called["ok"] = True
        # args is argparse.Namespace
        return 0

    monkeypatch.setattr(cli.help_workflow, "run_main", _stub_run_main)

    monkeypatch.setattr(sys, "argv", ["reaxkit", "help"])
    rc = cli.main()
    assert rc == 0
    assert called["ok"] is True


def test_intspec_kind_runs_without_task(monkeypatch: pytest.MonkeyPatch):
    # Stub introspection_workflow.run_main so we don't inspect filesystem/modules here.
    called = {"ok": False, "file": None, "folder": None}

    def _stub_run_main(file, folder):
        called["ok"] = True
        called["file"] = file
        called["folder"] = folder
        return 0

    monkeypatch.setattr(cli.introspection_workflow, "run_main", _stub_run_main)

    monkeypatch.setattr(sys, "argv", ["reaxkit", "intspec", "--file", "fort7_analyzer"])
    rc = cli.main()
    assert rc == 0
    assert called["ok"] is True
    assert called["file"] == "fort7_analyzer"
    assert called["folder"] is None


def test_preinject_noop_when_not_defaultable():
    # DEFAULTABLE is empty in cli.py currently; _preinject should not change argv.
    argv = ["reaxkit", "fort7", "get", "--file", "fort.7"]
    assert cli._preinject(argv) == argv
