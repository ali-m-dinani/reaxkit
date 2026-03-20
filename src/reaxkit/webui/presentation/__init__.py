"""Dash-specific Plotly presentation renderers."""

from reaxkit.webui.presentation.registry import render_figure
from reaxkit.webui.presentation.perf_config import load_ui_performance_config, ui_performance_config_path

__all__ = ["render_figure", "load_ui_performance_config", "ui_performance_config_path"]
