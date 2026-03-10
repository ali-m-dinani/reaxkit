"""Plotly renderer for shared ``single_plot`` specs."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from reaxkit.presentation.specs import PresentationSpec


def render_single_plot(
    rows: list[dict[str, Any]],
    *,
    spec: PresentationSpec,
    x_col: str | None = None,
    y_col: str | None = None,
    group_col: str | None = None,
) -> go.Figure:
    fig = go.Figure()
    if not rows:
        fig.update_layout(title="No plottable data")
        return fig

    cols = [str(col) for col in rows[0].keys()]
    numeric_cols = _numeric_columns(rows)
    use_x = x_col or spec.mapping.get("x_col") or ("iter" if "iter" in cols else ("frame_index" if "frame_index" in cols else (cols[0] if cols else "")))
    use_y = y_col or spec.mapping.get("y_col")
    if not use_y:
        use_y = next((col for col in numeric_cols if col != use_x), numeric_cols[0] if numeric_cols else None)
    use_group = group_col or spec.mapping.get("group_by_col") or ""
    if not use_y:
        fig.update_layout(title="No Y column selected")
        return fig

    groups: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        xv, yv = row.get(use_x), row.get(use_y)
        if xv is None or yv is None:
            continue
        try:
            xvf = float(xv)
            yvf = float(yv)
        except Exception:
            continue
        key = str(row.get(use_group, "all")) if use_group else "all"
        groups.setdefault(key, []).append((xvf, yvf))

    if not groups:
        fig.update_layout(title="No plottable numeric columns")
        return fig

    for key, points in groups.items():
        points.sort(key=lambda t: t[0])
        fig.add_trace(go.Scatter(x=[p[0] for p in points], y=[p[1] for p in points], mode="lines", name=key))

    title = str(spec.options.get("title") or f"{use_y} vs {use_x}")
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title=str(spec.options.get("xlabel") or use_x),
        yaxis_title=str(spec.options.get("ylabel") or use_y),
        showlegend=bool(spec.options.get("legend", bool(use_group))),
    )
    return fig


def _numeric_columns(rows: list[dict[str, Any]], *, sample_size: int = 50) -> list[str]:
    if not rows:
        return []
    cols = [str(col) for col in rows[0].keys()]
    sample = rows[: max(1, int(sample_size))]
    out: list[str] = []
    for col in cols:
        for row in sample:
            val = row.get(col)
            if isinstance(val, bool):
                continue
            if isinstance(val, (int, float)):
                out.append(col)
                break
    return out
