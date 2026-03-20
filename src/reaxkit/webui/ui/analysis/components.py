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
                className="rk-canvas-loading",
                children=html.Div(id="canvas-content", className="rk-canvas-box"),
            ),
            html.Div(id="canvas-export-status", className="rk-log-name"),
        ],
        className="rk-canvas-wrap",
    )


def dataset_info_panel() -> html.Div:
    """Footer dataset metadata area."""
    return html.Div(id="dataset-info-content", children="No dataset loaded.")
