"""Composed application layout for the ReaxKit web UI."""

from __future__ import annotations

from dash import dcc, html

from reaxkit.webui.ui.analysis.components import (
    dataset_info_panel,
    pipeline_controls,
    properties_panel,
    visualization_canvas,
)
from reaxkit.webui.runtime_paths import default_workspace_dir_for_dataset
from reaxkit.webui.ui.logs.components import log_page_panel
from reaxkit.webui.ui.shell.components import topbar
from reaxkit.webui.ui.shared.workspace import separator, activity_drawer


def build_layout() -> html.Div:
    """Construct the Dash layout shell."""
    return html.Div(
        [
            dcc.Interval(id="app-init", n_intervals=0, interval=50, max_intervals=1),
            dcc.Store(id="session-store", storage_type='session'),
            dcc.Store(id="pipeline-store"),
            dcc.Store(id="result-store"),
            dcc.Store(id="job-seen-store", data=[]),
            dcc.Store(id='workspace-state'),
            dcc.Store(id='properties-render-key'),
            dcc.Store(id='activity-view-key'),
            dcc.Interval(id='activity-tick', interval=1500, n_intervals=0),
            dcc.Interval(id="job-poll", interval=750, disabled=True),
            dcc.Store(id="viz-row-filters-store", data={"node_id": "", "rows": []}),
            dcc.Store(id="selected-curve-store", data={}),
            dcc.Store(id="ui-store", data={"page": "analysis", "help_open": False}),
            dcc.Store(
                id="config-store",
                data={
                    "dataset_path": ".",
                    "engine_name": "autodetect",
                    "manual_roles": [],
                    "role_xmolout": "xmolout",
                    "workspace_default": True,
                    "workspace_dir": str(default_workspace_dir_for_dataset(".")),
                    "draft_viz_type": "plot2d",
                },
            ),
            html.Div(topbar(), className="rk-panel rk-top"),
            html.Div([
                html.Aside([
                    html.Div(pipeline_controls(), id="panel-left", className="rk-panel rk-left"),
                    separator('hierarchy', 'horizontal', 'panel-left panel-props'),
                    html.Div(properties_panel(), id="panel-props", className="rk-panel rk-props"),
                ], id='workspace-sidebar', **{'aria-label': 'Pipeline and parameters'}),
                separator('sidebar', 'vertical', 'workspace-sidebar workspace-main'),
                html.Main([
                    html.Div(visualization_canvas(), id="panel-canvas", className="rk-panel rk-canvas"),
                    separator('drawer', 'horizontal', 'panel-canvas panel-drawer'),
                    activity_drawer(html.Div(log_page_panel(), id='panel-log-page')),
                ], id='workspace-main'),
            ], id='workspace-body'),
            html.Div(dataset_info_panel(), id="panel-info", className="rk-panel rk-info"),
            html.Div(id='workspace-announcement', className='rk-sr-only', role='status', **{'aria-live': 'polite'}),
        ],
        id='rk-workspace', className="rk-grid",
    )
