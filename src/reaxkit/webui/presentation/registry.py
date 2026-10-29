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
    filter_column = spec.options.get("filter_column")
    if filter_column:
        rows = [row for row in rows if row.get(filter_column) == spec.options.get("filter_value")]
    if renderer == "kymograph":
        from reaxkit.presentation.kymograph import kymograph_grid

        if not rows:
            return go.Figure().update_layout(title="No plottable data")
        coordinates, radii, values = kymograph_grid(
            rows, x_col or spec.mapping.get("x_col", "frame_index"),
            y_col or spec.mapping.get("y_col", "r"), color_col or spec.mapping.get("color_col", "g"),
        )
        figure = go.Figure(go.Heatmap(
            x=coordinates, y=radii, z=values, colorscale="Viridis",
            zmin=spec.options.get("vmin", 0.0), zmax=spec.options.get("vmax"),
            colorbar={"title": spec.options.get("colorbar_label", "g(r)")},
        ))
        figure.update_layout(template="plotly_white", title=spec.options.get("title", "RDF evolution"),
                             xaxis_title=spec.options.get("xlabel", "Frame index"), yaxis_title=spec.options.get("ylabel", "r (Å)"))
        return figure
    if renderer == "single_plot":
        return render_single_plot(rows, spec=spec, x_col=x_col, y_col=y_col, group_col=group_col)
    if renderer in {"scatter3d_points", "scatter3d"}:
        return render_scatter3d(rows, spec=spec, x_col=x_col, y_col=y_col, z_col=z_col, color_col=color_col)
    if renderer == "histogram":
        return render_histogram(rows, spec=spec, value_col=y_col or x_col)
    return None
