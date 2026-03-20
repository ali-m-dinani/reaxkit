"""Analysis page components."""

from __future__ import annotations

from dash import dcc, html


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
            html.Div(
                [
                    html.H3("Visualization Canvas"),
                    html.Div(
                        [
                            html.Button("Save", id="btn-canvas-primary", n_clicks=0, style={"display": "none"}),
                            html.Button("Save As", id="btn-canvas-secondary", n_clicks=0, style={"display": "none"}),
                        ],
                        className="rk-canvas-actions",
                    ),
                ],
                className="rk-canvas-head",
            ),
            dcc.Loading(
                id="canvas-content-loading",
                type="circle",
                children=html.Div(id="canvas-content", className="rk-canvas-box"),
            ),
            html.Div(id="canvas-export-status", className="rk-log-name"),
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
            dcc.Loading(
                id="result-content-loading",
                type="circle",
                children=html.Div(id="result-tab-content", className="rk-results-box"),
            ),
        ]
    )


def dataset_info_panel() -> html.Div:
    """Footer dataset metadata area."""
    return html.Div(id="dataset-info-content", children="No dataset loaded.")
