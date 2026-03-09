"""Reusable Dash UI components for the ReaxKit web interface."""

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
                    html.Span("Help", id="help-menu-trigger", className="rk-help-trigger"),
                    html.Div(
                        [
                            html.A("ReaxKit GitHub", href="https://github.com/ali-m-dinani/reaxkit?tab=readme-ov-file", target="_blank", className="rk-help-item"),
                            html.A("ReaxKit documentation", href="https://ali-m-dinani.github.io/reaxkit/", target="_blank", className="rk-help-item"),
                            html.A("UI documentation", href="https://ali-m-dinani.github.io/reaxkit/", target="_blank", className="rk-help-item"),
                            html.Button("check for updates", id="btn-help-check-updates", n_clicks=0, className="rk-help-item rk-help-btn"),
                            html.Div(id="help-update-status", className="rk-help-status"),
                        ],
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
                        children=html.Div(id="execute-loading-proxy", className="rk-spinner-anchor"),
                    ),
                    html.Span(id="status-banner", className="rk-badge"),
                ],
                className="rk-status-wrap",
            ),
        ],
        className="rk-topbar",
    )


def pipeline_controls() -> html.Div:
    """Dataset controls shown in the pipeline browser panel."""
    return html.Div(
        [
            html.H3("Pipeline Browser"),
            html.Div(id="pipeline-browser-tree", className="rk-tree"),
        ]
    )


def properties_panel() -> html.Div:
    """Properties panel placeholder for selected pipeline node."""
    return html.Div([html.H3("Parameters", id="parameters-title"), html.Div(id="properties-content")])


def visualization_canvas() -> html.Div:
    """Main canvas area driven by active result tab."""
    return html.Div(
        [
            html.H3("Visualization Canvas"),
            html.Div(id="canvas-content", className="rk-canvas-box"),
        ]
    )


def result_tabs() -> html.Div:
    """Result tab scaffold for Phase 1."""
    return html.Div(
        [
            html.H3("Result Tabs"),
            dcc.Tabs(
                id="result-tabs",
                value=None,
                children=[],
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Filter column"),
                            dcc.Dropdown(id="table-filter-col", options=[], value=None, clearable=True),
                            html.Label("Filter value"),
                            dcc.Input(id="table-filter-value", type="text", value=""),
                        ],
                        id="table-controls",
                        className="rk-inline-actions",
                    ),
                    html.Div(
                        [
                            html.Label("X"),
                            dcc.Dropdown(id="plot-x-col", options=[], value=None, clearable=False),
                            html.Label("Y"),
                            dcc.Dropdown(id="plot-y-col", options=[], value=None, clearable=False),
                            html.Label("Group"),
                            dcc.Dropdown(id="plot-group-col", options=[], value=None, clearable=True),
                        ],
                        id="plot-controls",
                        className="rk-inline-actions",
                    ),
                    html.Div(
                        [
                            html.Label("Histogram value"),
                            dcc.Dropdown(id="hist-col", options=[], value=None, clearable=False),
                        ],
                        id="hist-controls",
                        className="rk-inline-actions",
                    ),
                    html.Div(
                        [
                            html.Label("3D X"),
                            dcc.Dropdown(id="view3d-x-col", options=[], value=None, clearable=False),
                            html.Label("3D Y"),
                            dcc.Dropdown(id="view3d-y-col", options=[], value=None, clearable=False),
                            html.Label("3D Z"),
                            dcc.Dropdown(id="view3d-z-col", options=[], value=None, clearable=False),
                            html.Label("3D Color"),
                            dcc.Dropdown(id="view3d-color-col", options=[], value=None, clearable=True),
                        ],
                        id="view3d-controls",
                        className="rk-inline-actions",
                    ),
                    html.Div(
                        [
                            html.Label("Focus atom"),
                            dcc.Dropdown(id="focus-atom", options=[], value=None, clearable=True),
                        ],
                        id="sync-controls",
                        className="rk-inline-actions",
                    ),
                ]
            ),
            html.Div(id="result-tab-content", className="rk-results-box"),
        ]
    )


def dataset_info_panel() -> html.Div:
    """Footer dataset metadata area."""
    return html.Div(id="dataset-info-content", children="No dataset loaded.")


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
