"""Log page callbacks and helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dash import Input, Output, State, no_update


def _resolve_logs_root(config: dict[str, Any] | None) -> Path:
    cfg = dict(config or {})
    workspace_dir = str(cfg.get("workspace_dir") or "").strip()
    candidates: list[Path] = []
    if workspace_dir:
        path = Path(workspace_dir)
        candidates.append(path)
        if not path.is_absolute():
            candidates.append((Path.cwd() / path).resolve())
            candidates.append((Path(__file__).resolve().parents[2] / path).resolve())
    candidates.extend(
        [
            Path("reaxkit_workkspace"),
            Path("reaxkit_workspace"),
            Path.cwd() / "reaxkit_workkspace",
            Path.cwd() / "reaxkit_workspace",
            Path(__file__).resolve().parents[2] / "reaxkit_workkspace",
            Path(__file__).resolve().parents[2] / "reaxkit_workspace",
        ]
    )
    seen: set[str] = set()
    deduped: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    for root in deduped:
        logs = root / "logs"
        if logs.exists() and logs.is_dir():
            return logs
    return (deduped[0] / "logs") if deduped else (Path.cwd() / "reaxkit_workkspace" / "logs")


def _pick_latest(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.exists() and path.is_file()]
    if not existing:
        return None
    existing.sort(key=lambda path: path.stat().st_mtime)
    return existing[-1]


def _read_tail(path: Path | None, *, max_lines: int = 400) -> str:
    if path is None or not path.exists():
        return "No log file found."
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Failed to read log file: {exc}"
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return "\n".join(lines) if lines else "(empty log)"


def _select_log_files(config: dict[str, Any] | None) -> tuple[Path | None, Path | None]:
    logs_root = _resolve_logs_root(config)
    human_candidates = sorted(logs_root.glob("run_*.timing.log"))
    human = _pick_latest(human_candidates) or _pick_latest([logs_root / "timing_human.log"])

    low = _pick_latest([logs_root / "timing.log"])
    if low is not None:
        return human, low
    run_low_candidates = [path for path in logs_root.glob("run_*.log") if not path.name.endswith(".timing.log")]
    run_low_candidates.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0.0)
    low = _pick_latest(run_low_candidates)
    if low is None:
        low = _pick_latest([logs_root / "reaxkit.log"])
    return human, low


def register_log_callbacks(app) -> None:
    """Register callbacks for the log page."""

    @app.callback(
        Output("log-human-name", "children"),
        Output("log-human-content", "children"),
        Output("log-low-name", "children"),
        Output("log-low-content", "children"),
        Input("ui-store", "data"),
        Input("pipeline-store", "data"),
        State("config-store", "data"),
        prevent_initial_call=False,
    )
    def render_log_page(
        ui_data: dict[str, Any] | None,
        _snapshot: dict[str, Any] | None,
        config_data: dict[str, Any] | None,
    ):
        page = str((ui_data or {}).get("page") or "analysis").lower()
        if page != "log":
            return no_update, no_update, no_update, no_update
        human, low = _select_log_files(config_data)
        human_name = human.name if human else "(missing)"
        low_name = low.name if low else "(missing)"
        human_content = _read_tail(human)
        low_content = _read_tail(low)
        return human_name, human_content, low_name, low_content
