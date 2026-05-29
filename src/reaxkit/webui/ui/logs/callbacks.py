"""Log page callbacks and helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dash import Input, Output, no_update
from reaxkit.webui.runtime_paths import default_workspace_dir_for_dataset


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
            default_workspace_dir_for_dataset(str(Path.cwd())),
            default_workspace_dir_for_dataset("."),
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
    return (deduped[0] / "logs") if deduped else (default_workspace_dir_for_dataset(str(Path.cwd())) / "logs")


def _read_tail(path: Path | None, *, max_lines: int = 400, newest_first: bool = True) -> str:
    if path is None or not path.exists():
        return "No log file found."
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Failed to read log file: {exc}"
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    if newest_first:
        lines = list(reversed(lines))
    return "\n".join(lines) if lines else "(empty log)"


def _select_log_files(config: dict[str, Any] | None) -> tuple[Path | None, Path | None]:
    logs_root = _resolve_logs_root(config)
    general = logs_root / "general" / "reaxkit_general.log"
    timing = logs_root / "timing" / "human_readable_timing.log"
    return general, timing


def register_log_callbacks(app) -> None:
    """Register callbacks for the log page."""

    @app.callback(
        Output("log-human-name", "children"),
        Output("log-human-content", "children"),
        Output("log-low-name", "children"),
        Output("log-low-content", "children"),
        Input("ui-store", "data"),
        Input("pipeline-store", "data"),
        Input("log-refresh-tick", "n_intervals"),
        Input("config-store", "data"),
        prevent_initial_call=False,
    )
    def render_log_page(
        ui_data: dict[str, Any] | None,
        _snapshot: dict[str, Any] | None,
        _tick: int,
        config_input: dict[str, Any] | None,
    ):
        page = str((ui_data or {}).get("page") or "analysis").lower()
        if page != "log":
            return no_update, no_update, no_update, no_update
        general, timing = _select_log_files(config_input if isinstance(config_input, dict) else None)
        human_name = str(general) if general else "(missing)"
        low_name = str(timing) if timing else "(missing)"
        human_content = _read_tail(general, newest_first=True)
        low_content = _read_tail(timing, newest_first=True)
        return human_name, human_content, low_name, low_content
