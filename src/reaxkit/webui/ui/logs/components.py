"""Log page components."""

from __future__ import annotations

from dash import dcc, html


def log_page_panel() -> html.Div:
    """Log page content showing General and Timing logs."""
    return html.Div(
        [
            dcc.Interval(id="log-refresh-tick", interval=1500, n_intervals=0),
            html.H3("Log"),
            html.Div(
                [
                    html.H4("General log"),
                    html.Div(id="log-human-name", className="rk-log-name"),
                    html.Pre(id="log-human-content", className="rk-log-box"),
                ],
                className="rk-log-section",
            ),
            html.Div(
                [
                    html.H4("Timing log"),
                    html.Div(id="log-low-name", className="rk-log-name"),
                    html.Pre(id="log-low-content", className="rk-log-box"),
                ],
                className="rk-log-section",
            ),
        ],
        id="log-page-content",
        className="rk-log-page",
    )
