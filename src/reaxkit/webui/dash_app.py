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
    """Create and configure Dash app for Phase 2 shell."""
    Dash = _dash_imports()
    from reaxkit.webui.backend.api import WebUIApiService
    from reaxkit.webui.callbacks import register_callbacks
    from reaxkit.webui.ui.layout import build_layout
    from reaxkit.webui.ui.shared.styles import _CSS

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
        <style>{_CSS}</style>
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
    register_callbacks(app, service)
    return app
