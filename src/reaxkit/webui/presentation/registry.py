"""Renderer registry for Dash/Plotly figures."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from reaxkit.presentation.specs import ensure_presentation_spec
from reaxkit.webui.presentation.renderers.histogram import render_histogram
from reaxkit.webui.presentation.renderers.scatter3d import render_scatter3d
from reaxkit.webui.presentation.renderers.single import render_single_plot


def render_figure(
    rows: list[dict[str, Any]],
    *,
    presentation: dict[str, Any] | None = None,
    x_col: str | None = None,
    y_col: str | None = None,
    z_col: str | None = None,
    color_col: str | None = None,
    group_col: str | None = None,
) -> go.Figure | None:
    """Build a Plotly figure from shared presentation spec."""
    spec = ensure_presentation_spec(presentation or {})
    if spec is None:
        return None
    renderer = str(spec.renderer).lower()
    if renderer == "single_plot":
        return render_single_plot(rows, spec=spec, x_col=x_col, y_col=y_col, group_col=group_col)
    if renderer in {"scatter3d_points", "scatter3d"}:
        return render_scatter3d(rows, spec=spec, x_col=x_col, y_col=y_col, z_col=z_col, color_col=color_col)
    if renderer == "histogram":
        return render_histogram(rows, spec=spec, value_col=y_col or x_col)
    return None
