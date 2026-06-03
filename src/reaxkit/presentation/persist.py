"""
Persistence helpers for analysis results.

**Usage context**

- Import these helpers from presentation workflows that produce tables, files, or plots.
- Reuse the public APIs here to keep output formatting and artifact behavior consistent.
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reaxkit.core.platform.log import get_logger
from reaxkit.core.runtime.provenance import (
    effective_settings_from_args,
    runtime_metadata_from_args,
    user_settings_from_args,
)
from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout

logger = get_logger(__name__)


def _result_frames(result: Any) -> dict[str, pd.DataFrame]:
    """
    Result frames.
    """
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
    """
    Write csvs.
    """
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
    """
    Write numpy artifacts.
    """
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
    """
    Write figure artifacts.
    """
    if str(command) in {"active_site_structural", "get_active_site_structural"}:
        table = getattr(result, "table", None)
        if not isinstance(table, pd.DataFrame):
            return []
        try:
            from reaxkit.presentation.active_sites.plot_exports import save_structural_figures_tract_style
        except Exception as exc:  # pragma: no cover - defensive: plotting deps may be unavailable
            logger.debug("Skipping active-site structural figure export: %s", exc)
            return []
        return save_structural_figures_tract_style(table, out_dir, stem=str(analysis_id))

    if str(command) in {"active_site_events", "get_active_site_events"} and bool(
        getattr(result, "summary", {}).get("diagnostic", False)
        if isinstance(getattr(result, "summary", None), dict)
        else False
    ):
        try:
            from reaxkit.presentation.active_sites.plot_exports import save_event_diagnostic_figures
        except Exception as exc:  # pragma: no cover - defensive: plotting deps may be unavailable
            logger.debug("Skipping active-site event diagnostic figure export: %s", exc)
            return []
        return save_event_diagnostic_figures(
            getattr(result, "distance_table", pd.DataFrame()),
            getattr(result, "episode_table", pd.DataFrame()),
            getattr(result, "summary", {}),
            out_dir,
        )

    return []


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _result_request_value(result: Any, args: Any, name: str, default: Any = None) -> Any:
    request = getattr(result, "request", None)
    if request is not None and hasattr(request, name):
        return getattr(request, name)
    if hasattr(args, name):
        return getattr(args, name)
    return default


def _write_active_site_events_summary(command: str, out_dir: Path, result: Any, args: Any) -> list[str]:
    """Write TRACT-style active-site event summary text."""
    if str(command) not in {"active_site_events", "get_active_site_events"}:
        return []
    summary = getattr(result, "summary", None)
    if isinstance(summary, dict) and bool(summary.get("diagnostic", False)):
        return []
    table = getattr(result, "table", None)
    if not isinstance(table, pd.DataFrame):
        return []

    n_carbon = _as_int((summary or {}).get("n_carbon") if isinstance(summary, dict) else None, len(table))
    frames_analyzed = _as_int((summary or {}).get("frames_analyzed") if isinstance(summary, dict) else None, 0)
    stride = _as_int((summary or {}).get("every") if isinstance(summary, dict) else None, _result_request_value(result, args, "every", 1))
    persist = _as_int((summary or {}).get("persist") if isinstance(summary, dict) else None, _result_request_value(result, args, "persist", 1))
    timestep_fs = _as_float(_result_request_value(result, args, "timestep_fs", 10.0), 10.0)
    persist_fs = persist * max(1, stride) * timestep_fs

    r_co = _as_float(
        (summary or {}).get("r_CO") if isinstance(summary, dict) else None,
        _as_float(_result_request_value(result, args, "r_CO", _result_request_value(result, args, "r_co", 1.65)), 1.65),
    )
    r_csi = _as_float(
        (summary or {}).get("r_CSi") if isinstance(summary, dict) else None,
        _as_float(_result_request_value(result, args, "r_CSi", _result_request_value(result, args, "r_csi", 2.10)), 2.10),
    )

    total_o = _as_int(table["n_events_O"].sum() if "n_events_O" in table.columns else 0)
    total_si = _as_int(table["n_events_Si"].sum() if "n_events_Si" in table.columns else 0)
    reactive_o = _as_int(table["is_reactive_O"].sum() if "is_reactive_O" in table.columns else 0)
    reactive_si = _as_int(table["is_reactive_Si"].sum() if "is_reactive_Si" in table.columns else 0)
    pct_o = (100.0 * reactive_o / n_carbon) if n_carbon > 0 else 0.0
    pct_si = (100.0 * reactive_si / n_carbon) if n_carbon > 0 else 0.0
    input_name = Path(str(getattr(args, "input", "") or "")).name or str(getattr(args, "input", "") or "")

    lines = [
        "TRACT Tool 2 - Binding Event Summary",
        "=====================================",
        f"Input:          {input_name}",
        f"Frames analyzed:{frames_analyzed}  (stride={stride})",
        f"C atoms:        {n_carbon}",
        "",
        "Parameters:",
        f"  r_CO       = {r_co:g} A",
        f"  r_CSi      = {r_csi:g} A",
        f"  persist    = {persist} frames = {persist_fs:g} fs",
        f"  stride     = {stride} (every {stride}th raw frame analyzed)",
        "",
        "C-O binding events:",
        f"  Total confirmed:  {total_o}",
        f"  Reactive C atoms: {reactive_o} / {n_carbon}  ({pct_o:.2f}%)",
        "",
        "C-Si binding events:",
        f"  Total confirmed:  {total_si}",
        f"  Reactive C atoms: {reactive_si} / {n_carbon}  ({pct_si:.2f}%)",
        "",
        "Merge with Tool 1 output on column: atom_id",
        "",
    ]
    path = out_dir / "summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return [path.name]


def persist_analysis_result(command: str, result: Any, args: Any, *, write_csv: bool = True) -> Path:
    """
    Persist analysis result metadata (and optional CSV) under analysis/<command>/<run_id>/.
    
    This function is part of the ReaxKit presentation API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    command : str
        Input parameter used by this function.
    result : Any
        Input parameter used by this function.
    args : Any
        Input parameter used by this function.
    write_csv : bool, optional
        Input parameter used by this function.
    
    Returns
    -----
    Path
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.presentation.persist import persist_analysis_result
    result = persist_analysis_result(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
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
    text_files = _write_active_site_events_summary(command, out_dir, result, args)

    settings = {
        "command": str(command),
        "analysis_id": str(analysis_id),
        "run_id": getattr(args, "run_id", None),
        "parsed_id": getattr(args, "_parsed_id", None),
        "user_settings": user_settings_from_args(args),
        "effective_settings": effective_settings_from_args(args),
        "runtime": runtime_metadata_from_args(args),
        "artifacts": {
            "csv": csv_files,
            "npy": npy_files,
            "figures": figure_files,
            "text": text_files,
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
    """
    Append optional artifact names to existing analysis settings metadata.
    
    This function is part of the ReaxKit presentation API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    out_dir : Path
        Input parameter used by this function.
    reports : list[str] | None, optional
        Input parameter used by this function.
    
    Returns
    -----
    None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.presentation.persist import append_artifacts_to_settings
    result = append_artifacts_to_settings(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
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
