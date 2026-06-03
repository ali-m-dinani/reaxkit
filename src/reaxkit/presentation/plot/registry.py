"""
Renderer registry and dispatch API.

**Usage context**

- Import these helpers from presentation workflows that produce tables, files, or plots.
- Reuse the public APIs here to keep output formatting and artifact behavior consistent.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from reaxkit.presentation.plot.renderers.directed import DirectedPlotRenderer
from reaxkit.presentation.plot.renderers.dual_yaxis import DualYaxisPlotRenderer
from reaxkit.presentation.plot.renderers.errorbar import ErrorbarPlotRenderer
from reaxkit.presentation.plot.renderers.heatmap2d import Heatmap2DRenderer
from reaxkit.presentation.plot.renderers.multi_subplots import MultiSubplotsRenderer
from reaxkit.presentation.plot.renderers.scatter3d import Scatter3DRenderer
from reaxkit.presentation.plot.renderers.single import SinglePlotRenderer
from reaxkit.presentation.plot.renderers.tornado import TornadoPlotRenderer
from reaxkit.presentation.plot.renderers.boxplot import BoxWhiskerPlotRenderer
from reaxkit.presentation.plot.renderers.beeswarm import BeeswarmPlotRenderer
from reaxkit.presentation.plot.renderers.wireframe3d import Wireframe3DRenderer
from reaxkit.presentation.plot.renderers.wireframe3d_subplots import Wireframe3DSubplotsRenderer

PLOT_REGISTRY = {
    "single_plot": SinglePlotRenderer(),
    "directed_plot": DirectedPlotRenderer(),
    "dual_yaxis_plot": DualYaxisPlotRenderer(),
    "multi_subplots": MultiSubplotsRenderer(),
    "tornado_plot": TornadoPlotRenderer(),
    "scatter3d_points": Scatter3DRenderer(),
    "wireframe3d_plot": Wireframe3DRenderer(),
    "wireframe3d_subplots": Wireframe3DSubplotsRenderer(),
    "heatmap2d_from_3d": Heatmap2DRenderer(),
    "errorbar_plot": ErrorbarPlotRenderer(),
    "box_whisker_plot": BoxWhiskerPlotRenderer(),
    "beeswarm_plot": BeeswarmPlotRenderer(),
    # backward-compatible aliases
    "line": SinglePlotRenderer(),
    "multicurve": SinglePlotRenderer(),
    "dual_axis": DualYaxisPlotRenderer(),
    "scatter3d": Scatter3DRenderer(),
    "wireframe3d": Wireframe3DRenderer(),
    "heatmap2d": Heatmap2DRenderer(),
    "errorbar": ErrorbarPlotRenderer(),
    "boxplot": BoxWhiskerPlotRenderer(),
    "beeswarm": BeeswarmPlotRenderer(),
}


def _plot_type_of(result: Any) -> str:
    """
    Plot type of.
    """
    if isinstance(result, dict):
        p = result.get("plot_type")
    else:
        p = getattr(result, "plot_type", None)
    if not p:
        raise ValueError("result must include 'plot_type'")
    return str(p)


def plot(result: Any, style: Optional[Mapping[str, Any]] = None):
    """
    Dispatch to renderer by ``result.plot_type``.
    
    This function is part of the ReaxKit presentation API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    result : Any
        Input parameter used by this function.
    style : Optional[Mapping[str, Any]], optional
        Input parameter used by this function.
    
    Returns
    -----
    Any
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.presentation.plot.registry import plot
    result = plot(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    ptype = _plot_type_of(result)
    renderer = PLOT_REGISTRY.get(ptype)
    if renderer is None:
        raise KeyError(f"Unknown plot_type '{ptype}'. Available: {sorted(PLOT_REGISTRY.keys())}")
    return renderer.render(result, style)
