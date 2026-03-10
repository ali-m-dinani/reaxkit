"""Renderer registry for Dash/Plotly figures."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from reaxkit.presentation.specs import PresentationSpec, ensure_presentation_spec
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
    view_type: str | None = None,
) -> go.Figure | None:
    """Build a Plotly figure from shared presentation spec."""
    spec = ensure_presentation_spec(presentation or {})
    if spec is None:
        vtype = str(view_type or "").strip().lower()
        if vtype in {"plot", "plot2d", "single_plot"}:
            spec = PresentationSpec(
                renderer="single_plot",
                label="Plot",
                mapping={
                    "x_col": str(x_col or ""),
                    "y_col": str(y_col or ""),
                    "group_by_col": str(group_col or ""),
                },
                options={},
                view_type="plot2d",
            )
        elif vtype in {"hist", "histogram"}:
            spec = PresentationSpec(
                renderer="histogram",
                label="Histogram",
                mapping={"value_col": str(y_col or x_col or "")},
                options={},
                view_type="histogram",
            )
        elif vtype in {"scatter", "scatter3d", "3d"}:
            spec = PresentationSpec(
                renderer="scatter3d_points",
                label="3D",
                mapping={
                    "x_col": str(x_col or ""),
                    "y_col": str(y_col or ""),
                    "z_col": str(z_col or ""),
                    "color_col": str(color_col or ""),
                },
                options={},
                view_type="scatter3d",
            )
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
