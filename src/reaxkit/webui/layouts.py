"""Dash layout composition for ReaxKit Web UI."""

from __future__ import annotations

from dash import dcc, html

from reaxkit.webui.components import (
    dataset_info_panel,
    pipeline_controls,
    properties_panel,
    result_tabs,
    topbar,
    visualization_canvas,
)


_CSS = """
body { margin: 0; font-family: Segoe UI, Tahoma, sans-serif; background: #edf3f7; }
.rk-grid {
  display: grid; height: 100vh; gap: 8px; padding: 8px;
  grid-template-columns: 320px 1fr;
  grid-template-rows: 56px 1fr 260px 42px;
  grid-template-areas:
    "top top"
    "left canvas"
    "props results"
    "info info";
}
.rk-panel { background: #fff; border: 1px solid #cedae3; border-radius: 10px; padding: 10px; overflow: auto; }
.rk-top { grid-area: top; }
.rk-left { grid-area: left; }
.rk-canvas { grid-area: canvas; }
.rk-props { grid-area: props; }
.rk-results { grid-area: results; }
.rk-info { grid-area: info; }
.rk-topbar { display: flex; align-items: center; gap: 16px; }
.rk-status-wrap { margin-left: auto; display: flex; align-items: center; justify-content: flex-end; min-width: 120px; }
.rk-badge { margin-left: auto; background: #e5f2fb; border: 1px solid #b6d2e8; border-radius: 14px; padding: 2px 8px; font-size: 12px; }
.rk-badge-error { margin-left: auto; background: #fde8e8; border: 1px solid #f1b5b5; color: #8a1c1c; border-radius: 14px; padding: 2px 8px; font-size: 12px; }
.rk-badge-warn { margin-left: auto; background: #fff5dc; border: 1px solid #f0d18d; color: #7a4f00; border-radius: 14px; padding: 2px 8px; font-size: 12px; }
.rk-stack { display: grid; gap: 8px; margin-bottom: 10px; }
.rk-subtitle { margin: 8px 0 8px; font-size: 14px; color: #38536a; }
.rk-inline-actions { display: grid; grid-template-columns: repeat(6, minmax(90px, auto)); gap: 8px; align-items: center; margin: 8px 0; }
.rk-btn-save {
  width: 180px;
  height: 30px;
}
.rk-btn-exec {
  width: 90px;
  height: 30px;
}
.rk-tree {
  border: 1px solid #d4e0e8;
  background: #f9fcff;
  border-radius: 8px;
  padding: 8px;
  min-height: 190px;
  max-height: 250px;
  overflow: auto;
  display: grid;
  gap: 2px;
  margin-bottom: 10px;
}
.rk-tree-node {
  border: 1px solid transparent; border-radius: 4px; background: transparent;
  text-align: left; padding: 4px 8px; display: flex; align-items: center; gap: 8px; cursor: pointer;
  font-family: "Segoe UI", Tahoma, sans-serif;
}
.rk-tree-node:hover { background: #eef5fb; border-color: #d4e3f0; }
.rk-tree-node.selected { background: #d8e7f8; border-color: #b8cfe8; }
.rk-tree-prefix { color: #70879c; font-family: Consolas, "Courier New", monospace; white-space: pre; }
.rk-tree-icon { width: 16px; }
.rk-tree-label { font-weight: 600; color: #2f4a63; }
.rk-tree-status { margin-left: auto; color: #61778a; font-size: 12px; }
.rk-tree-meta { color: #3f5d74; padding: 2px 8px; display: flex; align-items: center; gap: 8px; font-family: "Segoe UI", Tahoma, sans-serif; }
.rk-tree-empty { color: #607788; font-size: 13px; }
.rk-help-inline { display: inline-flex; align-items: center; gap: 6px; }
.rk-help-dot {
  display: inline-flex; align-items: center; justify-content: center;
  width: 16px; height: 16px; border-radius: 50%;
  border: 1px solid #9cb2c3; color: #3d5568; font-size: 11px; font-weight: 700;
  cursor: help; background: #f2f7fb;
}
.rk-canvas-box, .rk-results-box { border: 1px dashed #bfd0de; border-radius: 8px; min-height: 140px; padding: 10px; }
@media (max-width: 980px) {
  .rk-grid {
    grid-template-columns: 1fr;
    grid-template-rows: 56px 280px 250px 260px 220px 42px;
    grid-template-areas:
      "top"
      "left"
      "props"
      "canvas"
      "results"
      "info";
  }
}
"""


def build_layout() -> html.Div:
    """Construct the ParaView-inspired Dash layout shell."""
    return html.Div(
        [
            dcc.Interval(id="app-init", n_intervals=0, interval=50, max_intervals=1),
            dcc.Store(id="session-store"),
            dcc.Store(id="pipeline-store"),
            dcc.Store(id="result-store"),
            dcc.Store(
                id="config-store",
                data={
                    "dataset_path": ".",
                    "engine_name": "autodetect",
                    "manual_roles": [],
                    "role_xmolout": "xmolout",
                },
            ),
            html.Div(topbar(), className="rk-panel rk-top"),
            html.Div(pipeline_controls(), className="rk-panel rk-left"),
            html.Div(visualization_canvas(), className="rk-panel rk-canvas"),
            html.Div(properties_panel(), className="rk-panel rk-props"),
            html.Div(result_tabs(), className="rk-panel rk-results"),
            html.Div(dataset_info_panel(), className="rk-panel rk-info"),
        ],
        className="rk-grid",
    )
