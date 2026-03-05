"""Persistence helpers for analysis results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from reaxkit.core.log import get_logger
from reaxkit.core.storage_layout import ReaxkitStorageLayout

logger = get_logger(__name__)


def _result_frames(result: Any) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    if hasattr(result, "table") and isinstance(getattr(result, "table"), pd.DataFrame):
        frames["table"] = getattr(result, "table")
    if is_dataclass(result):
        for f in fields(result):
            value = getattr(result, f.name)
            if isinstance(value, pd.DataFrame):
                frames[f.name] = value
    return frames


def _write_csvs(out_dir: Path, result: Any) -> list[str]:
    frames = _result_frames(result)
    written: list[str] = []
    if not frames:
        return written
    if set(frames.keys()) == {"table"}:
        path = out_dir / "result.csv"
        frames["table"].to_csv(path, index=False)
        return [path.name]
    for key, frame in frames.items():
        safe = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in key).strip("_") or "table"
        path = out_dir / f"{safe}.csv"
        frame.to_csv(path, index=False)
        written.append(path.name)
    return written


def persist_analysis_result(command: str, result: Any, args: Any) -> Path:
    """Persist analysis result to CSV + JSON under analysis/<command>/<analysis_id>/."""
    project_root = Path(getattr(args, "project_root", "."))
    analysis_id = (
        getattr(args, "analysis_id", None)
        or getattr(args, "_analysis_id", None)
        or getattr(args, "run_id", None)
        or "analysis"
    )
    layout = ReaxkitStorageLayout(project_root=project_root)
    out_dir = layout.analysis_root / str(command) / str(analysis_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = _write_csvs(out_dir, result)

    settings = {
        "command": str(command),
        "analysis_id": str(analysis_id),
        "run_id": getattr(args, "run_id", None),
        "parsed_id": getattr(args, "_parsed_id", None),
        "artifacts": {
            "csv": csv_files,
            "settings": "settings.json",
        },
    }
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")
    return out_dir
