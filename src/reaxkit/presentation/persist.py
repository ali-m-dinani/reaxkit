"""Persistence helpers for analysis results."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
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


def _write_numpy_artifacts(command: str, out_dir: Path, result: Any) -> list[str]:
    written: list[str] = []
    if str(command) != "active_site_structural":
        return written
    soap = getattr(result, "soap_descriptors", None)
    if isinstance(soap, np.ndarray):
        path = out_dir / "soap_descriptors.npy"
        np.save(path, soap)
        written.append(path.name)
    return written


def _write_figure_artifacts(command: str, out_dir: Path, result: Any, *, analysis_id: str) -> list[str]:
    if str(command) != "active_site_structural":
        return []
    table = getattr(result, "table", None)
    if not isinstance(table, pd.DataFrame):
        return []
    try:
        from reaxkit.analysis.active_sites.plot_exports import save_structural_figures_tract_style
    except Exception as exc:  # pragma: no cover - defensive: plotting deps may be unavailable
        logger.debug("Skipping active-site structural figure export: %s", exc)
        return []
    return save_structural_figures_tract_style(table, out_dir, stem=str(analysis_id))


def persist_analysis_result(command: str, result: Any, args: Any, *, write_csv: bool = True) -> Path:
    """Persist analysis result metadata (and optional CSV) under analysis/<command>/<run_id>/."""
    project_root = Path(getattr(args, "project_root", "."))
    analysis_id = (
        getattr(args, "analysis_id", None)
        or getattr(args, "run_id", None)
        or getattr(args, "_analysis_id", None)
        or "analysis"
    )
    layout = ReaxkitStorageLayout(project_root=project_root)
    out_dir = layout.analysis_root / str(command) / str(analysis_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = _write_csvs(out_dir, result) if write_csv else []
    npy_files = _write_numpy_artifacts(command, out_dir, result)
    figure_files = _write_figure_artifacts(command, out_dir, result, analysis_id=str(analysis_id))

    settings = {
        "command": str(command),
        "analysis_id": str(analysis_id),
        "run_id": getattr(args, "run_id", None),
        "parsed_id": getattr(args, "_parsed_id", None),
        "artifacts": {
            "csv": csv_files,
            "npy": npy_files,
            "figures": figure_files,
            "settings": "settings.json",
        },
    }
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")
    return out_dir


def append_artifacts_to_settings(
    out_dir: Path,
    *,
    reports: list[str] | None = None,
) -> None:
    """Append optional artifact names to existing analysis settings metadata."""
    settings_path = out_dir / "settings.json"
    if not settings_path.exists():
        return
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception:
        return

    artifacts = payload.setdefault("artifacts", {})
    if reports:
        existing = list(artifacts.get("reports") or [])
        seen = set(existing)
        for item in reports:
            name = str(item)
            if name and name not in seen:
                existing.append(name)
                seen.add(name)
        artifacts["reports"] = existing

    settings_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
