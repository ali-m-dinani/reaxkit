"""Callback registration composition for the ReaxKit web UI."""

from __future__ import annotations

from reaxkit.webui.backend.api import WebUIApiService
from reaxkit.webui.ui.analysis.callbacks import register_analysis_callbacks
from reaxkit.webui.ui.logs.callbacks import register_log_callbacks
from reaxkit.webui.ui.shell.callbacks import register_shell_callbacks


def register_callbacks(app, service: WebUIApiService) -> None:
    """Register all Dash callbacks for the web UI."""
    register_shell_callbacks(app, service)
    register_log_callbacks(app)
    register_analysis_callbacks(app, service)


__all__ = ["register_callbacks"]
