"""Web UI entrypoints for Dash-based ReaxKit interface."""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import tempfile
import time
from typing import Any
from pathlib import Path

from reaxkit.webui.backend.api import WebUIApiService
from reaxkit.webui.dash_app import create_dash_app


def _console_logging_enabled() -> bool:
    raw = str(os.environ.get("REAXKIT_WEBUI_LOG_CONSOLE", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _single_instance_enabled() -> bool:
    raw = str(os.environ.get("REAXKIT_WEBUI_SINGLE_INSTANCE", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _instance_lock_path(port: int) -> Path:
    return Path(tempfile.gettempdir()) / f"reaxkit_webui_{port}.pid"


def _read_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        pid = int(text)
        if pid <= 0:
            return None
        return pid
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _terminate_pid(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    os.kill(pid, signal.SIGTERM)


def _ensure_single_instance(port: int, logger: logging.Logger) -> None:
    if not _single_instance_enabled():
        logger.info("Single-instance guard disabled by REAXKIT_WEBUI_SINGLE_INSTANCE")
        return
    lock_path = _instance_lock_path(port)
    existing_pid = _read_pid(lock_path)
    current_pid = os.getpid()
    if existing_pid and existing_pid != current_pid and _pid_alive(existing_pid):
        logger.warning(
            "WebUI single-instance guard: terminating existing process pid=%s on port=%s",
            existing_pid,
            port,
        )
        try:
            _terminate_pid(existing_pid)
            deadline = time.time() + 4.0
            while time.time() < deadline:
                if not _pid_alive(existing_pid):
                    break
                time.sleep(0.1)
            if _pid_alive(existing_pid):
                logger.warning(
                    "WebUI single-instance guard: process pid=%s still alive after terminate attempt",
                    existing_pid,
                )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "WebUI single-instance guard: failed terminating pid=%s error=%s",
                existing_pid,
                exc,
            )
    try:
        lock_path.write_text(str(current_pid), encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        logger.warning("WebUI single-instance guard: failed writing lock %s error=%s", lock_path, exc)
        return

    def _cleanup_lock() -> None:
        try:
            if _read_pid(lock_path) == current_pid:
                lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    atexit.register(_cleanup_lock)


def create_service() -> WebUIApiService:
    """Create the in-process Web UI API service."""
    return WebUIApiService()


def create_app():
    """Create a WSGI server app for deployment."""
    dash_app = create_dash_app()
    return dash_app.server


def _log_analysis_dropdown_wiring(dash_app: Any) -> None:
    logger = logging.getLogger(__name__)
    try:
        callback_map = getattr(dash_app, "callback_map", {}) or {}
        hits: list[tuple[str, list[str]]] = []
        for out_key, meta in callback_map.items():
            inputs = meta.get("inputs", []) if isinstance(meta, dict) else []
            input_props: list[str] = []
            for inp in inputs:
                if not isinstance(inp, dict):
                    continue
                in_id = inp.get("id")
                in_prop = inp.get("property")
                input_props.append(f"{in_id}.{in_prop}")
            out_text = str(out_key)
            joined_inputs = " | ".join(input_props)
            if "input-analysis-type" in out_text or "input-analysis-type" in joined_inputs:
                hits.append((out_text, input_props))
        if not hits:
            logger.info("UI callback wiring: no callback references input-analysis-type")
            return
        logger.info("UI callback wiring for input-analysis-type:")
        for out_text, input_props in hits:
            logger.info("  OUT %s <= %s", out_text, input_props)
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to inspect callback wiring: %s", exc)


def main() -> None:
    """Run Dash dev server."""
    # Verbose terminal logging for Web UI debugging.
    level_name = str(os.environ.get("REAXKIT_WEBUI_LOG_LEVEL", "DEBUG")).upper()
    level = getattr(logging, level_name, logging.DEBUG)
    console_log_enabled = False  # Set to False to silence terminal logs.
    if console_log_enabled:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            force=True,
        )
    else:
        # Keep logging calls safe but silence console output.
        logging.basicConfig(
            level=logging.CRITICAL + 1,
            handlers=[logging.NullHandler()],
            force=True,
        )
    for name in ("reaxkit", "reaxkit.webui", "dash", "flask", "werkzeug"):
        logging.getLogger(name).setLevel(level)
    logging.getLogger(__name__).info("WebUI startup app_module=%s cwd=%s", __file__, os.getcwd())
    port_raw = str(os.environ.get("REAXKIT_WEBUI_PORT", "8060")).strip()
    try:
        port = int(port_raw)
    except Exception:
        port = 8060
    logging.getLogger(__name__).info("WebUI startup pid=%s port=%s", os.getpid(), port)
    _ensure_single_instance(port, logging.getLogger(__name__))
    # Always write trace output to a single workspace path, independent of launch cwd.
    project_root = Path(__file__).resolve().parents[3]
    trace_target = project_root / "reaxkit_workspace" / "log" / "UI" / "ui_trace.log"
    try:
        trace_target.parent.mkdir(parents=True, exist_ok=True)
        with trace_target.open("a", encoding="utf-8") as fh:
            fh.write(f"[UI_TRACE] startup app_module={__file__} cwd={os.getcwd()}\n")
        logging.getLogger(__name__).info("UI trace file target=%s", trace_target)
    except Exception as exc:  # pragma: no cover
        logging.getLogger(__name__).warning("UI trace file write failed target=%s error=%s", trace_target, exc)

    dash_app = create_dash_app()
    if str(os.environ.get("REAXKIT_WEBUI_LOG_CALLBACKS", "1")).strip().lower() not in {"0", "false", "no"}:
        _log_analysis_dropdown_wiring(dash_app)
    dash_app.run(
        debug=True,
        use_reloader=False,
        dev_tools_silence_routes_logging=False,
        port=port,
    )


if __name__ == "__main__":
    main()
