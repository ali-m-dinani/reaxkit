"""Log page components."""

from __future__ import annotations

from dash import html


def log_page_panel() -> html.Div:
    """Log page content showing human-readable and lower-level logs."""
    return html.Div(
        [
            html.H3("Log"),
            html.Div(
                [
                    html.H4("human-readable log"),
                    html.Div(id="log-human-name", className="rk-log-name"),
                    html.Pre(id="log-human-content", className="rk-log-box"),
                ],
                className="rk-log-section",
            ),
            html.Div(
                [
                    html.H4("lower-level log"),
                    html.Div(id="log-low-name", className="rk-log-name"),
                    html.Pre(id="log-low-content", className="rk-log-box"),
                ],
                className="rk-log-section",
            ),
        ],
        id="log-page-content",
        className="rk-log-page",
    )
