"""Web UI entrypoints for Dash-based ReaxKit interface."""

from __future__ import annotations

from reaxkit.webui.backend.api import WebUIApiService
from reaxkit.webui.dash_app import create_dash_app


def create_service() -> WebUIApiService:
    """Create the in-process Web UI API service."""
    return WebUIApiService()


def create_app():
    """Create a WSGI server app for deployment."""
    dash_app = create_dash_app()
    return dash_app.server


def main() -> None:
    """Run Dash dev server."""
    dash_app = create_dash_app()
    dash_app.run(debug=True)


if __name__ == "__main__":
    main()
