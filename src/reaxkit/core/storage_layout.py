"""Run-scoped storage layout utilities.

This module implements the project layout:
  inputs/<run_id>/
  data/raw/<run_id>/
  data/parsed/<parsed_id>/
  data/run_index/<run_id>.json
  analysis/<workflow>/<analysis_id>/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _safe_run_id(run_id: str) -> str:
    value = str(run_id or "").strip()
    if not value:
        raise ValueError("run_id cannot be empty.")
    return value


def default_project_root() -> Path:
    candidate = Path("reaxkit")
    return candidate if candidate.exists() and candidate.is_dir() else Path(".")


def generate_run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    token = secrets.token_hex(3)
    return f"run_{stamp}_{token}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def directory_fingerprint(root: Path) -> str:
    if not root.exists():
        return "missing"
    files = [p for p in root.rglob("*") if p.is_file()]
    if not files:
        return "empty"
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: str(p.relative_to(root)).replace("\\", "/")):
        rel = str(path.relative_to(root)).replace("\\", "/")
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class ReaxkitStorageLayout:
    project_root: Path = Path(".")

    @property
    def inputs_root(self) -> Path:
        return self.project_root / "inputs"

    @property
    def data_root(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_root(self) -> Path:
        return self.data_root / "raw"

    @property
    def parsed_root(self) -> Path:
        return self.data_root / "parsed"

    @property
    def run_index_root(self) -> Path:
        return self.data_root / "run_index"

    @property
    def parsed_index_path(self) -> Path:
        return self.parsed_root / "index.json"

    @property
    def analysis_root(self) -> Path:
        return self.project_root / "analysis"

    @property
    def figures_root(self) -> Path:
        return self.project_root / "figures"

    @property
    def reports_root(self) -> Path:
        return self.project_root / "reports"

    @property
    def cache_root(self) -> Path:
        return self.project_root / "cache"

    @property
    def logs_root(self) -> Path:
        return self.project_root / "logs"

    def ensure_base_layout(self) -> None:
        for root in (
            self.inputs_root,
            self.raw_root,
            self.parsed_root,
            self.run_index_root,
            self.analysis_root,
            self.figures_root,
            self.reports_root,
            self.cache_root,
            self.logs_root,
        ):
            root.mkdir(parents=True, exist_ok=True)

    def input_run_dir(self, run_id: str) -> Path:
        return self.inputs_root / _safe_run_id(run_id)

    def raw_run_dir(self, run_id: str) -> Path:
        return self.raw_root / _safe_run_id(run_id)

    def run_index_path(self, run_id: str) -> Path:
        return self.run_index_root / f"{_safe_run_id(run_id)}.json"

    def parsed_dir(self, parsed_id: str) -> Path:
        return self.parsed_root / str(parsed_id)

    def parsed_run_dir(self, run_id: str) -> Path:
        return self.parsed_root / _safe_run_id(run_id)

    def ensure_run_layout(self, run_id: str) -> None:
        self.ensure_base_layout()
        self.input_run_dir(run_id).mkdir(parents=True, exist_ok=True)
        self.raw_run_dir(run_id).mkdir(parents=True, exist_ok=True)
        self.parsed_run_dir(run_id).mkdir(parents=True, exist_ok=True)

    def register_parsed_dataset(
        self,
        *,
        run_id: str,
        parser_version: str,
        engine: str,
    ) -> str:
        run_id = _safe_run_id(run_id)
        self.ensure_run_layout(run_id)
        raw_dir = self.raw_run_dir(run_id)
        raw_hash = directory_fingerprint(raw_dir)
        parsed_id = run_id
        parsed_dir = self.parsed_run_dir(run_id)
        parsed_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            parsed_dir / "meta.json",
            {
                "parsed_id": parsed_id,
                "run_id": run_id,
                "raw_hash": raw_hash,
                "engine": engine,
                "parser_version": parser_version,
                "updated_at": _utc_now_iso(),
            },
        )
        _write_json(
            self.run_index_path(run_id),
            {
                "run_id": run_id,
                "raw_dir": str(raw_dir),
                "raw_hash": raw_hash,
                "parsed_id": parsed_id,
                "engine": engine,
                "parser_version": parser_version,
                "updated_at": _utc_now_iso(),
            },
        )
        return parsed_id


def add_storage_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", default=None, help="Run identifier for run-scoped layout (e.g., run_91ac0e).")
    parser.add_argument(
        "--project-root",
        default=str(default_project_root()),
        help="Project root that contains inputs/, data/, analysis/, etc.",
    )
    parser.add_argument("--analysis-id", default=None, help="Optional analysis artifact id; defaults to run id.")


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists() or dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dst)


def _snapshot_raw_from_source(out: dict, raw_dir: Path) -> None:
    detection_keys = ("input", "run_dir", "xmolout", "fort7", "summary")
    source = None
    for key in detection_keys:
        value = out.get(key)
        if not value:
            continue
        path = Path(str(value))
        if path.exists():
            source = path if path.is_dir() else path.parent
            break
    if source is None:
        source = Path(".")
    if source.resolve() == raw_dir.resolve():
        return

    for name in (
        "xmolout",
        "summary.txt",
        "fort.7",
        "fort.13",
        "fort.57",
        "fort.73",
        "fort.74",
        "fort.76",
        "fort.78",
        "fort.79",
        "fort.99",
        "trainset.in",
        "params",
        "control",
        "eregime.in",
        "vels",
        "molfra.out",
        "ffield",
        "dump.lammpstrj",
        "lammpstrj",
        "log.lammps",
    ):
        _copy_if_exists(source / name, raw_dir / name)


def normalize_storage_args(args: dict) -> dict:
    out = dict(args)
    run_id = out.get("run_id")
    if not run_id:
        run_id = generate_run_id()
        out["run_id"] = run_id

    project_root = Path(out.get("project_root") or default_project_root())
    out["project_root"] = str(project_root)
    layout = ReaxkitStorageLayout(project_root=project_root)
    layout.ensure_run_layout(str(run_id))
    raw_dir = layout.raw_run_dir(str(run_id))
    _snapshot_raw_from_source(out, raw_dir)

    if not out.get("input") or str(out.get("input")) == ".":
        out["input"] = str(raw_dir)
    if not out.get("run_dir") or str(out.get("run_dir")) == ".":
        out["run_dir"] = str(raw_dir)
    if not out.get("cache_dir"):
        out["cache_dir"] = str(layout.cache_root / str(run_id))

    # Rewrite default bare filenames to the run-scoped raw directory.
    default_files = {
        "xmolout": "xmolout",
        "summary": "summary.txt",
        "fort7": "fort.7",
        "fort13": "fort.13",
        "fort57": "fort.57",
        "fort73": "fort.73",
        "fort74": "fort.74",
        "fort76": "fort.76",
        "fort78": "fort.78",
        "fort79": "fort.79",
        "fort99": "fort.99",
        "trainset": "trainset.in",
        "params": "params",
        "control": "control",
        "eregime": "eregime.in",
        "vels": "vels",
        "molfra": "molfra.out",
        "ffield": "ffield",
    }
    for key, default_name in default_files.items():
        raw = out.get(key)
        if not raw:
            continue
        path = Path(str(raw))
        if path.is_absolute():
            continue
        if path.parent != Path("."):
            continue
        if path.name != default_name:
            continue
        out[key] = str(raw_dir / default_name)
    return out
