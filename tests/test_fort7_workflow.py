"""
Tests for fort7_workflow.

These tests focus on workflow wiring and task behavior using monkeypatch stubs,
so they do not require real fort.7 files or heavy analyzers.

Validates:
- register_tasks() creates expected subcommands and attaches _run
- _task_get() summary mode exports CSV
- _task_edges() exports CSV when edges returned
- _task_constats() exports CSV when stats returned
- _task_bond_ts() exports CSV (tidy mode)
- _task_bond_events() exports CSV when events returned
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest

import reaxkit.workflows.per_file.fort7_workflow as wf


# -----------------------------
# CLI registration
# -----------------------------

def test_register_tasks_wires_subcommands():
    parser = argparse.ArgumentParser(prog="reaxkit fort7")
    subs = parser.add_subparsers(dest="cmd", required=True)

    wf.register_tasks(subs)

    # Ensure each command exists and has a callable _run
    for cmd in ["get", "edges", "constats", "bond-ts", "bond-events"]:
        ns = parser.parse_args([cmd, "--file", "fort.7", "--yaxis", "charge"] if cmd == "get" else [cmd, "--file", "fort.7"])
        assert hasattr(ns, "_run")
        assert callable(getattr(ns, "_run"))


# -----------------------------
# Task stubs
# -----------------------------

class _DummyFort7Handler:
    def __init__(self, file):
        self.file = file
        self.num_atoms = 3
        self._frames = []  # only used in an internal recovery branch


def test_task_get_summary_exports_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Stub handler + analyzer
    monkeypatch.setattr(wf, "Fort7Handler", _DummyFort7Handler)

    def _stub_get_summaries(handler, feat, frames=None, regex=False, add_index_cols=True):
        return pd.DataFrame(
            {
                "frame_idx": [0, 1],
                "iter": [10, 11],
                "total_BO": [1.0, 1.5],
            }
        )

    monkeypatch.setattr(wf, "get_fort7_data_summaries", _stub_get_summaries)

    # Make resolve_output_path deterministic
    def _stub_resolve(path, workflow_name):
        return Path(path)

    monkeypatch.setattr(wf, "resolve_output_path", _stub_resolve)

    args = argparse.Namespace(
        kind="fort7",
        file=str(tmp_path / "fort.7"),
        yaxis="total_BO",
        atom=None,
        frames=None,
        xaxis="iter",
        control="control",
        regex=False,
        export=str(tmp_path / "out.csv"),
        save=None,
        plot=False,
    )

    rc = wf._task_get(args)
    assert rc == 0
    out = Path(args.export)
    assert out.exists()

    df = pd.read_csv(out)
    assert list(df.columns) == ["iter", "total_BO"]
    assert df["total_BO"].tolist() == [1.0, 1.5]


def test_task_edges_exports_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(wf, "Fort7Handler", _DummyFort7Handler)

    def _stub_edges(*args, **kwargs):
        return pd.DataFrame(
            {
                "frame_idx": [0, 0, 1],
                "iter": [10, 10, 11],
                "src": [1, 1, 2],
                "dst": [2, 3, 3],
                "bo": [0.6, 0.4, 0.5],
            }
        )

    monkeypatch.setattr(wf, "connection_list", _stub_edges)
    monkeypatch.setattr(wf, "resolve_output_path", lambda p, wn: Path(p))

    args = argparse.Namespace(
        kind="fort7",
        file=str(tmp_path / "fort.7"),
        frames=None,
        min_bo=0.3,
        directed=False,
        aggregate="max",
        include_self=False,
        xaxis="frame",
        control="control",
        export=str(tmp_path / "edges.csv"),
        save=None,
        plot=False,
    )

    rc = wf._task_edges(args)
    assert rc == 0
    assert Path(args.export).exists()


def test_task_constats_exports_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(wf, "Fort7Handler", _DummyFort7Handler)

    def _stub_stats(*args, **kwargs):
        return pd.DataFrame(
            {
                "frame_idx": [0, 1],
                "iter": [10, 11],
                "value": [2.0, 3.0],
            }
        )

    monkeypatch.setattr(wf, "connection_stats_over_frames", _stub_stats)
    monkeypatch.setattr(wf, "resolve_output_path", lambda p, wn: Path(p))

    args = argparse.Namespace(
        kind="fort7",
        file=str(tmp_path / "fort.7"),
        frames=None,
        min_bo=0.3,
        directed=False,
        how="mean",
        export=str(tmp_path / "stats.csv"),
        save=None,
    )

    rc = wf._task_constats(args)
    assert rc == 0
    assert Path(args.export).exists()


def test_task_bond_ts_exports_csv_tidy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(wf, "Fort7Handler", _DummyFort7Handler)

    def _stub_ts(*args, **kwargs):
        return pd.DataFrame(
            {
                "frame_idx": [0, 1],
                "iter": [10, 11],
                "src": [1, 1],
                "dst": [2, 2],
                "bo": [0.6, 0.2],
            }
        )

    monkeypatch.setattr(wf, "bond_timeseries", _stub_ts)
    monkeypatch.setattr(wf, "resolve_output_path", lambda p, wn: Path(p))

    args = argparse.Namespace(
        kind="fort7",
        file=str(tmp_path / "fort.7"),
        frames=None,
        directed=False,
        bo_threshold=0.0,
        wide=False,
        xaxis="iter",
        control="control",
        src=None,
        dst=None,
        export=str(tmp_path / "bo.csv"),
        save=None,
        plot=False,
    )

    rc = wf._task_bond_ts(args)
    assert rc == 0
    assert Path(args.export).exists()

    df = pd.read_csv(args.export)
    assert set(["frame_idx", "iter", "src", "dst", "bo"]).issubset(df.columns)


def test_task_bond_events_exports_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(wf, "Fort7Handler", _DummyFort7Handler)

    def _stub_events(*args, **kwargs):
        return pd.DataFrame(
            {
                "src": [1],
                "dst": [2],
                "event": ["formed"],
                "frame_idx": [5],
                "iter": [50],
            }
        )

    monkeypatch.setattr(wf, "bond_events", _stub_events)
    monkeypatch.setattr(wf, "resolve_output_path", lambda p, wn: Path(p))

    args = argparse.Namespace(
        kind="fort7",
        file=str(tmp_path / "fort.7"),
        frames=None,
        src=1,
        dst=2,
        threshold=0.35,
        hysteresis=0.05,
        smooth="ma",
        window=7,
        ema_alpha=None,
        min_run=3,
        xaxis="iter",
        directed=False,
        export=str(tmp_path / "events.csv"),
        save=None,
        plot=False,
    )

    rc = wf._task_bond_events(args)
    assert rc == 0
    assert Path(args.export).exists()
