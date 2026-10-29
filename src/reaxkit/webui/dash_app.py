"""Dash application factory for ReaxKit Web UI."""

from __future__ import annotations


def _dash_imports():
    try:
        from dash import Dash
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Dash is not installed. Install with: pip install dash"
        ) from exc
    return Dash


def create_dash_app():
    """Create the responsive scientific workspace and its shared backend."""
    Dash = _dash_imports()
    from reaxkit.webui.backend.api import WebUIApiService
    from reaxkit.webui.callbacks import register_callbacks
    from reaxkit.webui.ui.shared.debounce import enable_input_debounce
    from reaxkit.webui.ui.layout import build_layout

    enable_input_debounce()

    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        title="ReaxKit GUI",
    )
    app.index_string = f"""<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>"""
    app.layout = build_layout()

    service = WebUIApiService()
    app.reaxkit_service = service
    from flask import g, request
    from time import perf_counter

    @app.server.before_request
    def record_start():
        g.reaxkit_started = perf_counter()

    @app.server.after_request
    def record_response(response):
        elapsed = (perf_counter() - g.reaxkit_started) * 1000
        response.headers['Server-Timing'] = f'reaxkit;dur={elapsed:.3f}'
        if request.path == '/_dash-update-component':
            service.metrics.record('dash_callback', elapsed, response_bytes=response.content_length or 0)
        return response
    register_callbacks(app, service)
    return app
