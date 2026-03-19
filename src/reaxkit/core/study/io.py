"""Study IO/status helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from reaxkit.core.study.logging import utc_now


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def resolve_existing_file(primary: Path, *fallbacks: str) -> Path:
    if primary.exists():
        return primary
    for name in fallbacks:
        candidate = primary.parent / name
        if candidate.exists():
            return candidate
    return primary


def write_named_status(
    *,
    study_root: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    csv_name: str,
    json_name: str,
) -> tuple[Path, Path]:
    json_path = study_root / json_name
    csv_path = study_root / csv_name
    payload = {
        "generated_at_utc": utc_now(),
        "summary": summary,
        "rows": rows,
    }
    write_json(json_path, payload)

    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = []
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        if fieldnames:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in fieldnames})
        else:
            fh.write("")
    return csv_path, json_path


def write_study_run_status(
    *,
    study_root: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    run_status_json_file: str,
    run_status_csv_file: str,
) -> tuple[Path, Path]:
    status_dir = study_root
    status_dir.mkdir(parents=True, exist_ok=True)

    json_path = status_dir / run_status_json_file
    csv_path = status_dir / run_status_csv_file

    payload = {
        "generated_at_utc": utc_now(),
        "summary": summary,
        "rows": rows,
    }
    write_json(json_path, payload)

    fieldnames = [
        "case_id",
        "replicate_id",
        "status",
        "run",
        "done",
        "skip",
        "wait",
        "fail",
        "started_at",
        "finished_at",
        "duration_min",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    return json_path, csv_path


def run_stage_manifest_path(stage_dir: Path, *, run_stage_manifest_file: str, legacy_stage_manifest_file: str) -> Path:
    return resolve_existing_file(stage_dir / run_stage_manifest_file, legacy_stage_manifest_file)


def run_replicate_manifest_path(
    rep_dir: Path, *, run_replicate_manifest_file: str, legacy_replicate_manifest_file: str
) -> Path:
    return resolve_existing_file(rep_dir / run_replicate_manifest_file, legacy_replicate_manifest_file)


def run_case_manifest_path(case_dir: Path, *, run_case_manifest_file: str, legacy_case_manifest_file: str) -> Path:
    return resolve_existing_file(case_dir / run_case_manifest_file, legacy_case_manifest_file)


def run_status_csv_path(study_root: Path, *, run_status_csv_file: str, legacy_run_status_csv_file: str) -> Path:
    return resolve_existing_file(study_root / run_status_csv_file, legacy_run_status_csv_file)


def analysis_status_csv_path(study_root: Path, *, analysis_status_csv_file: str, legacy_analysis_status_csv_file: str) -> Path:
    return resolve_existing_file(study_root / analysis_status_csv_file, legacy_analysis_status_csv_file)


def analysis_status_json_path(
    study_root: Path, *, analysis_status_json_file: str, legacy_analysis_status_json_file: str
) -> Path:
    return resolve_existing_file(study_root / analysis_status_json_file, legacy_analysis_status_json_file)


def load_status_rows(csv_path: Path, *, not_found_message: str) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(not_found_message)
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def stage_status_path(stage_dir: Path, *, stage_status_file: str) -> Path:
    return stage_dir / stage_status_file


def load_stage_status(stage_dir: Path, *, stage_status_file: str) -> dict[str, Any]:
    path = stage_status_path(stage_dir, stage_status_file=stage_status_file)
    if not path.exists():
        return {"status": "pending", "jobs": [], "updated_at_utc": None}
    payload = read_json(path)
    payload.setdefault("jobs", [])
    payload.setdefault("status", "pending")
    return payload


def write_stage_status(stage_dir: Path, status: dict[str, Any], *, stage_status_file: str) -> None:
    payload = dict(status)
    payload["updated_at_utc"] = utc_now()
    write_json(stage_status_path(stage_dir, stage_status_file=stage_status_file), payload)


def analysis_status_path(stage_dir: Path, *, analysis_status_file: str) -> Path:
    return stage_dir / analysis_status_file


def load_analysis_status(stage_dir: Path, *, analysis_status_file: str) -> dict[str, Any]:
    path = analysis_status_path(stage_dir, analysis_status_file=analysis_status_file)
    if not path.exists():
        return {"analyses": {}, "updated_at_utc": None}
    payload = read_json(path)
    if not isinstance(payload.get("analyses"), dict):
        payload["analyses"] = {}
    return payload


def write_analysis_status(stage_dir: Path, status: dict[str, Any], *, analysis_status_file: str) -> None:
    payload = dict(status)
    payload["updated_at_utc"] = utc_now()
    write_json(analysis_status_path(stage_dir, analysis_status_file=analysis_status_file), payload)


def analysis_manifest_path(stage_dir: Path, *, analysis_manifest_file: str) -> Path:
    return stage_dir / analysis_manifest_file


def load_analysis_manifest(stage_dir: Path, *, analysis_manifest_file: str) -> dict[str, Any]:
    path = analysis_manifest_path(stage_dir, analysis_manifest_file=analysis_manifest_file)
    if not path.exists():
        return {"runs": [], "updated_at_utc": None}
    payload = read_json(path)
    runs = payload.get("runs")
    if not isinstance(runs, list):
        payload["runs"] = []
    return payload


def write_analysis_manifest(stage_dir: Path, payload: dict[str, Any], *, analysis_manifest_file: str) -> None:
    out = dict(payload)
    out["updated_at_utc"] = utc_now()
    write_json(analysis_manifest_path(stage_dir, analysis_manifest_file=analysis_manifest_file), out)

