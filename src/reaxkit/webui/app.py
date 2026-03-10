"""Web UI entrypoints for Dash-based ReaxKit interface."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any
from pathlib import Path

from reaxkit.webui.backend.api import WebUIApiService
from reaxkit.webui.dash_app import create_dash_app


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
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
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
    trace_env = str(os.environ.get("REAXKIT_UI_TRACE_PATH", "")).strip()
    trace_targets = []
    if trace_env:
        trace_targets.append(Path(trace_env))
    trace_targets.append(Path(os.getcwd()) / "ui_trace.log")
    trace_targets.append(Path(tempfile.gettempdir()) / "reaxkit_ui_trace.log")
    for target in trace_targets:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as fh:
                fh.write(f"[UI_TRACE] startup app_module={__file__} cwd={os.getcwd()}\n")
            logging.getLogger(__name__).info("UI trace file target=%s", target)
        except Exception as exc:  # pragma: no cover
            logging.getLogger(__name__).warning("UI trace file write failed target=%s error=%s", target, exc)

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
