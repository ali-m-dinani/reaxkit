"""Plotly renderer for histogram presentation specs."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from reaxkit.presentation.specs import PresentationSpec


def render_histogram(rows: list[dict[str, Any]], *, spec: PresentationSpec, value_col: str | None = None) -> go.Figure:
    fig = go.Figure()
    if not rows:
        fig.update_layout(title="No numeric data for histogram.")
        return fig
    use_col = value_col or spec.mapping.get("value_col") or _default_numeric_column(rows)
    vals = []
    for row in rows:
        try:
            vals.append(float(row.get(use_col)))
        except Exception:
            continue
    if not vals:
        fig.update_layout(title="No numeric data for histogram.")
        return fig
    fig.add_trace(go.Histogram(x=vals, nbinsx=40))
    fig.update_layout(template="plotly_white", title=str(spec.options.get("title") or f"{use_col} Distribution"))
    return fig


def _default_numeric_column(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "value"
    cols = [str(col) for col in rows[0].keys()]
    sample = rows[:50]
    for col in cols:
        for row in sample:
            val = row.get(col)
            if isinstance(val, bool):
                continue
            if isinstance(val, (int, float)):
                return col
    return cols[0] if cols else "value"
