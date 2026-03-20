"""Application shell components."""

from __future__ import annotations

from dash import dcc, html


def topbar() -> html.Div:
    """Application top navigation bar."""
    return html.Div(
        [
            html.Strong("ReaxKit GUI"),
            html.Button("Analysis", id="btn-nav-analysis", n_clicks=0, className="rk-nav-btn active"),
            html.Button("Log", id="btn-nav-log", n_clicks=0, className="rk-nav-btn"),
            html.Div(
                [
                    html.Button("Help", id="help-menu-trigger", n_clicks=0, className="rk-help-trigger"),
                    html.Div(
                        [
                            html.A("ReaxKit GitHub", href="https://github.com/ali-m-dinani/reaxkit?tab=readme-ov-file", target="_blank", className="rk-help-item"),
                            html.A("ReaxKit documentation", href="https://ali-m-dinani.github.io/reaxkit/", target="_blank", className="rk-help-item"),
                            html.A("UI documentation", href="https://ali-m-dinani.github.io/reaxkit/", target="_blank", className="rk-help-item"),
                            html.Button("check for updates", id="btn-help-check-updates", n_clicks=0, className="rk-help-item rk-help-btn"),
                            html.Div(id="help-update-status", className="rk-help-status"),
                        ],
                        id="help-menu-dropdown",
                        className="rk-help-dropdown",
                    ),
                ],
                className="rk-help-menu",
            ),
            html.Div(
                [
                    dcc.Loading(
                        id="execute-loading",
                        type="circle",
                        target_components={
                            "execute-loading-proxy": "children",
                            "canvas-content": "children",
                            "result-tab-content": "children",
                        },
                        children=html.Div(id="execute-loading-proxy", className="rk-spinner-anchor"),
                    ),
                    html.Span(id="status-banner", className="rk-badge"),
                ],
                className="rk-status-wrap",
            ),
        ],
        className="rk-topbar",
    )
