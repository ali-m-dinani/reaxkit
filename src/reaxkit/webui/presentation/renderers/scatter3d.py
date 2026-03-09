"""Plotly renderer for shared ``scatter3d_points`` specs."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from reaxkit.presentation.specs import PresentationSpec


def render_scatter3d(
    rows: list[dict[str, Any]],
    *,
    spec: PresentationSpec,
    x_col: str | None = None,
    y_col: str | None = None,
    z_col: str | None = None,
    color_col: str | None = None,
) -> go.Figure:
    fig = go.Figure()
    if not rows:
        fig.update_layout(title="No 3D data")
        return fig

    cols = list(rows[0].keys())
    use_x = x_col or spec.mapping.get("x_col") or ("x" if "x" in cols else ("iter" if "iter" in cols else "frame_index"))
    use_y = y_col or spec.mapping.get("y_col") or ("y" if "y" in cols else ("atom_id" if "atom_id" in cols else use_x))
    use_z = z_col or spec.mapping.get("z_col") or ("z" if "z" in cols else ("msd" if "msd" in cols else use_y))
    use_color = color_col or spec.mapping.get("color_col") or ""

    xvals: list[float] = []
    yvals: list[float] = []
    zvals: list[float] = []
    cvals: list[float] = []
    for row in rows:
        try:
            xv = float(row.get(use_x, 0.0))
            yv = float(row.get(use_y, 0.0))
            zv = float(row.get(use_z, 0.0))
        except Exception:
            continue
        xvals.append(xv)
        yvals.append(yv)
        zvals.append(zv)
        if use_color:
            try:
                cvals.append(float(row.get(use_color, 0.0)))
            except Exception:
                cvals.append(0.0)

    marker: dict[str, Any] = {"size": 4, "opacity": 0.8}
    if use_color:
        marker.update({"color": cvals, "colorscale": "Viridis", "colorbar": {"title": use_color}})

    fig.add_trace(go.Scatter3d(x=xvals, y=yvals, z=zvals, mode="markers", marker=marker, name="points"))
    fig.update_layout(
        template="plotly_white",
        title=str(spec.options.get("title") or f"3D View: {use_x}, {use_y}, {use_z}"),
        scene={"xaxis_title": use_x, "yaxis_title": use_y, "zaxis_title": use_z},
    )
    return fig
