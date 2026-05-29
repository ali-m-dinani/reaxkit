from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path

import pytest

path_cli = import_module("reaxkit.cli.path")
cli_main = import_module("reaxkit.cli.main")


def _make_run_dir(root: Path, name: str, mtime: float) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / "xmolout"
    marker.write_text("data", encoding="utf-8")
    os.utime(run_dir, (mtime, mtime))
    os.utime(marker, (mtime, mtime))
    return run_dir


def test_free_up_keep_last_deletes_older(tmp_path: Path):
    raw_root = tmp_path / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    _make_run_dir(raw_root, "run_old", 100)
    _make_run_dir(raw_root, "run_mid", 200)
    _make_run_dir(raw_root, "run_new", 300)

    deleted = path_cli.free_up_keep_last(raw_root, keep=2)

    assert [p.name for p in deleted] == ["run_old"]
    assert (raw_root / "run_old").exists() is False
    assert (raw_root / "run_mid").exists() is True
    assert (raw_root / "run_new").exists() is True


def test_free_up_compress_old_creates_archives(tmp_path: Path):
    raw_root = tmp_path / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    _make_run_dir(raw_root, "run_old", 100)
    _make_run_dir(raw_root, "run_mid", 200)
    _make_run_dir(raw_root, "run_new", 300)

    archives, skipped = path_cli.free_up_compress_old(raw_root, keep=1, compression="gz")

    assert skipped == []
    assert sorted(p.name for p in archives) == ["run_mid.tar.gz", "run_old.tar.gz"]
    assert (raw_root / "run_old").exists() is False
    assert (raw_root / "run_mid").exists() is False
    assert (raw_root / "run_new").exists() is True
    assert (raw_root / "run_old.tar.gz").exists() is True
    assert (raw_root / "run_mid.tar.gz").exists() is True


def test_free_up_compress_old_dry_run(tmp_path: Path):
    raw_root = tmp_path / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    _make_run_dir(raw_root, "run_old", 100)
    _make_run_dir(raw_root, "run_new", 300)

    archives, skipped = path_cli.free_up_compress_old(raw_root, keep=1, compression="gz", dry_run=True)

    assert skipped == []
    assert [p.name for p in archives] == ["run_old.tar.gz"]
    assert (raw_root / "run_old").exists() is True
    assert (raw_root / "run_old.tar.gz").exists() is False


def test_cli_free_up_command_routing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    _make_run_dir(raw_root, "run_1", 100)
    _make_run_dir(raw_root, "run_2", 200)

    monkeypatch.setattr(
        sys,
        "argv",
        ["reaxkit", "free-up", "--last", "1", "--raw-root", str(raw_root)],
    )
    rc = cli_main.main()

    assert rc == 0
    assert (raw_root / "run_1").exists() is False
    assert (raw_root / "run_2").exists() is True
