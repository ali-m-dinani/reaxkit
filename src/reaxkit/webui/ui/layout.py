"""Composed application layout for the ReaxKit web UI."""

from __future__ import annotations

from dash import dcc, html

from reaxkit.webui.ui.analysis.components import (
    dataset_info_panel,
    pipeline_controls,
    properties_panel,
    result_tabs,
    visualization_canvas,
)
from reaxkit.webui.ui.logs.components import log_page_panel
from reaxkit.webui.ui.shell.components import topbar


def build_layout() -> html.Div:
    """Construct the Dash layout shell."""
    return html.Div(
        [
            dcc.Interval(id="app-init", n_intervals=0, interval=50, max_intervals=1),
            dcc.Store(id="session-store"),
            dcc.Store(id="pipeline-store"),
            dcc.Store(id="result-store"),
            dcc.Store(id="selected-curve-store", data={}),
            dcc.Store(id="ui-store", data={"page": "analysis"}),
            dcc.Store(
                id="config-store",
                data={
                    "dataset_path": ".",
                    "engine_name": "autodetect",
                    "manual_roles": [],
                    "role_xmolout": "xmolout",
                    "workspace_default": True,
                    "workspace_dir": "reaxkit_workspace/",
                    "draft_viz_type": "plot2d",
                },
            ),
            html.Div(topbar(), className="rk-panel rk-top"),
            html.Div(pipeline_controls(), id="panel-left", className="rk-panel rk-left"),
            html.Div(visualization_canvas(), id="panel-canvas", className="rk-panel rk-canvas"),
            html.Div(properties_panel(), id="panel-props", className="rk-panel rk-props"),
            html.Div(result_tabs(), id="panel-results", className="rk-panel rk-results"),
            html.Div(dataset_info_panel(), id="panel-info", className="rk-panel rk-info"),
            html.Div(log_page_panel(), id="panel-log-page", className="rk-panel rk-page-full", style={"display": "none"}),
        ],
        className="rk-grid",
    )
