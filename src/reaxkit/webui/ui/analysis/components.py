"""Analysis page components."""

from __future__ import annotations

from dash import dcc, html
from reaxkit.webui.ui.shared.workspace import action, icon, empty_state


def pipeline_controls() -> html.Div:
    """Dataset controls shown in the pipeline browser panel."""
    return html.Div(
        [
            html.Div([icon('tree'), html.H3('Hierarchy'),
                action('Collapse or restore hierarchy', 'hierarchy', icon_name='collapse', **{'aria-expanded': 'true'})], className='rk-panel-head'),
            html.Div([html.Label('Find a node', htmlFor='hierarchy-search', className='rk-sr-only'),
                dcc.Input(id='hierarchy-search', type='search', placeholder='Find a node…', debounce=0.2)], className='rk-tree-search'),
            html.Div(id="pipeline-browser-tree", className="rk-tree", role='tree', **{'aria-label': 'Pipeline hierarchy'}),
            html.Div(html.Button('Delete analysis', id='btn-delete-analysis', n_clicks=0, disabled=True,
                title='Delete the selected analysis and its utilities, presentations, and results.'), className='rk-hierarchy-actions'),
            html.Small('Select a node to inspect its parameters.', className='rk-panel-hint'),
        ], className='rk-sidebar-section'
    )


def properties_panel() -> html.Div:
    """Properties panel placeholder for selected pipeline node."""
    return html.Div([
        html.Div([icon('parameters'), html.H3("Parameters", id="parameters-title"),
            action('Collapse or restore parameters', 'properties', icon_name='collapse', **{'aria-expanded': 'true'})], className='rk-panel-head'),
        html.Div(id='parameters-status', className='rk-panel-hint', role='status'),
        html.Div(id="properties-content"),
    ], className='rk-sidebar-section')


def visualization_canvas() -> html.Div:
    """Main canvas area driven by active result tab."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div([icon('plot'), html.Div([html.Small('PRESENTATION', className='rk-eyebrow'), html.H3('Workspace', id='canvas-title')])], className='rk-canvas-heading'),
                    html.Div(
                        [
                            html.Button("Save", id="btn-canvas-primary", n_clicks=0, style={"display": "none"}),
                            html.Button("Save As", id="btn-canvas-secondary", n_clicks=0, style={"display": "none"}),
                            action('Maximize or restore canvas', 'maximize', icon_name='expand', **{'aria-pressed': 'false'}),
                        ],
                        className="rk-canvas-actions",
                    ),
                ],
                className="rk-canvas-head",
            ),
            html.Small(id='canvas-preview-notice'),
            dcc.Loading(
                id="canvas-content-loading",
                type="circle",
                className="rk-canvas-loading",
                parent_className="rk-canvas-loading",
                parent_style={"display": "flex", "flex": "1 1 auto", "minHeight": "0", "width": "100%"},
                overlay_style={"visibility": "visible"},
                delay_show=200,
                style={"display": "flex", "flex": "1 1 auto", "minHeight": "0", "width": "100%"},
                children=html.Div(
                    id="canvas-content",
                    children=empty_state('Explore your simulation', 'Select Dataset in the hierarchy to open a system, then add an analysis and presentation.'),
                    className="rk-canvas-box",
                    style={"display": "flex", "flex": "1 1 auto", "minHeight": "0", "height": "100%"},
                ),
            ),
            html.Div(id="canvas-export-status", className="rk-log-name"),
        ],
        className="rk-canvas-wrap",
    )


def dataset_info_panel() -> html.Div:
    """Footer dataset metadata area."""
    return html.Div(id="dataset-info-content", children="No dataset loaded.")
