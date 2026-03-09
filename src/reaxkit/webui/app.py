"""Web UI entrypoints for Dash-based ReaxKit interface."""

from __future__ import annotations

import logging
from datetime import datetime

from reaxkit.webui.backend.api import WebUIApiService
from reaxkit.webui.dash_app import create_dash_app


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_info(message: str) -> None:
    print(f"[{_ts()}] INFO    {message}")


def log_warning(message: str) -> None:
    print(f"[{_ts()}] WARNING {message}")


def log_error(message: str) -> None:
    print(f"[{_ts()}] ERROR   {message}")


def _attach_console_logging(dash_app) -> None:
    """Attach request/error logging hooks once per Dash app instance."""
    server = dash_app.server
    if getattr(server, "_reaxkit_console_logging_attached", False):
        return

    @server.after_request
    def _log_response(response):  # pragma: no cover - runtime hook
        from flask import request

        code = int(response.status_code)
        if code >= 500:
            log_error(f"HTTP {code} {request.method} {request.path}")
        elif code >= 400:
            log_warning(f"HTTP {code} {request.method} {request.path}")
        return response

    @server.errorhandler(Exception)
    def _log_exception(exc):  # pragma: no cover - runtime hook
        log_error(f"Unhandled exception: {exc}")
        raise exc

    logging.getLogger("werkzeug").setLevel(logging.INFO)
    server._reaxkit_console_logging_attached = True
    log_info("Console logging hooks attached")


def create_service() -> WebUIApiService:
    """Create the in-process Web UI API service."""
    return WebUIApiService()


def create_app():
    """Create a WSGI server app for deployment."""
    dash_app = create_dash_app()
    _attach_console_logging(dash_app)
    return dash_app.server


def main() -> None:
    """Run Dash dev server."""
    dash_app = create_dash_app()
    _attach_console_logging(dash_app)
    log_info("Starting ReaxKit Dash dev server")
    dash_app.run(debug=True)


if __name__ == "__main__":
    main()
