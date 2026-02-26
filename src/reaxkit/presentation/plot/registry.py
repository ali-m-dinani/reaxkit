"""Renderer registry and dispatch API."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from reaxkit.presentation.plot.renderers.directed import DirectedPlotRenderer
from reaxkit.presentation.plot.renderers.dual_yaxis import DualYaxisPlotRenderer
from reaxkit.presentation.plot.renderers.heatmap2d import Heatmap2DRenderer
from reaxkit.presentation.plot.renderers.multi_subplots import MultiSubplotsRenderer
from reaxkit.presentation.plot.renderers.scatter3d import Scatter3DRenderer
from reaxkit.presentation.plot.renderers.single import SinglePlotRenderer
from reaxkit.presentation.plot.renderers.tornado import TornadoPlotRenderer

PLOT_REGISTRY = {
    "single_plot": SinglePlotRenderer(),
    "directed_plot": DirectedPlotRenderer(),
    "dual_yaxis_plot": DualYaxisPlotRenderer(),
    "multi_subplots": MultiSubplotsRenderer(),
    "tornado_plot": TornadoPlotRenderer(),
    "scatter3d_points": Scatter3DRenderer(),
    "heatmap2d_from_3d": Heatmap2DRenderer(),
    # backward-compatible aliases
    "line": SinglePlotRenderer(),
    "multicurve": SinglePlotRenderer(),
    "dual_axis": DualYaxisPlotRenderer(),
    "scatter3d": Scatter3DRenderer(),
    "heatmap2d": Heatmap2DRenderer(),
}


def _plot_type_of(result: Any) -> str:
    if isinstance(result, dict):
        p = result.get("plot_type")
    else:
        p = getattr(result, "plot_type", None)
    if not p:
        raise ValueError("result must include 'plot_type'")
    return str(p)


def plot(result: Any, style: Optional[Mapping[str, Any]] = None):
    """Dispatch to renderer by ``result.plot_type``."""
    ptype = _plot_type_of(result)
    renderer = PLOT_REGISTRY.get(ptype)
    if renderer is None:
        raise KeyError(f"Unknown plot_type '{ptype}'. Available: {sorted(PLOT_REGISTRY.keys())}")
    return renderer.render(result, style)
