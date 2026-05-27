"""Renderer-based plotting API."""

from reaxkit.presentation.plot.renderers.base import PlotRenderer
from reaxkit.presentation.plot.renderers.boxplot import BoxWhiskerPlotRenderer
from reaxkit.presentation.plot.renderers.beeswarm import BeeswarmPlotRenderer
from reaxkit.presentation.plot.renderers.directed import DirectedPlotRenderer
from reaxkit.presentation.plot.renderers.dual_yaxis import DualYaxisPlotRenderer
from reaxkit.presentation.plot.renderers.errorbar import ErrorbarPlotRenderer
from reaxkit.presentation.plot.renderers.heatmap2d import Heatmap2DRenderer
from reaxkit.presentation.plot.renderers.multi_subplots import MultiSubplotsRenderer
from reaxkit.presentation.plot.registry import PLOT_REGISTRY, plot
from reaxkit.presentation.plot.renderers.scatter3d import Scatter3DRenderer
from reaxkit.presentation.plot.renderers.single import SinglePlotRenderer
from reaxkit.presentation.plot.renderers.tornado import TornadoPlotRenderer
from reaxkit.presentation.plot.renderers.wireframe3d import Wireframe3DRenderer
from reaxkit.presentation.plot.renderers.wireframe3d_subplots import Wireframe3DSubplotsRenderer


def single_plot(x=None, y=None, **kwargs):
    return plot({"plot_type": "single_plot", "x": x, "y": y, **kwargs})


def directed_plot(x, y, **kwargs):
    return plot({"plot_type": "directed_plot", "x": x, "y": y, **kwargs})


def dual_yaxis_plot(x, y1, y2, **kwargs):
    return plot({"plot_type": "dual_yaxis_plot", "x": x, "y1": y1, "y2": y2, **kwargs})


def multi_subplots(subplots, **kwargs):
    return plot({"plot_type": "multi_subplots", "subplots": subplots, **kwargs})


def tornado_plot(labels, min_vals, max_vals, **kwargs):
    return plot({"plot_type": "tornado_plot", "labels": labels, "min_vals": min_vals, "max_vals": max_vals, **kwargs})


def scatter3d_points(coords, values, **kwargs):
    return plot({"plot_type": "scatter3d_points", "coords": coords, "values": values, **kwargs})


def heatmap2d_from_3d(coords, values, **kwargs):
    return plot({"plot_type": "heatmap2d_from_3d", "coords": coords, "values": values, **kwargs})


def errorbar_plot(x=None, y=None, yerr=None, **kwargs):
    return plot({"plot_type": "errorbar_plot", "x": x, "y": y, "yerr": yerr, **kwargs})


def box_whisker_plot(data, labels=None, **kwargs):
    return plot({"plot_type": "box_whisker_plot", "data": data, "labels": labels, **kwargs})


def beeswarm_plot(x, y, **kwargs):
    return plot({"plot_type": "beeswarm_plot", "x": x, "y": y, **kwargs})


__all__ = [
    "PlotRenderer",
    "SinglePlotRenderer",
    "DirectedPlotRenderer",
    "DualYaxisPlotRenderer",
    "MultiSubplotsRenderer",
    "TornadoPlotRenderer",
    "Scatter3DRenderer",
    "Wireframe3DRenderer",
    "Wireframe3DSubplotsRenderer",
    "Heatmap2DRenderer",
    "ErrorbarPlotRenderer",
    "BoxWhiskerPlotRenderer",
    "BeeswarmPlotRenderer",
    "PLOT_REGISTRY",
    "plot",
    "single_plot",
    "directed_plot",
    "dual_yaxis_plot",
    "multi_subplots",
    "tornado_plot",
    "scatter3d_points",
    "heatmap2d_from_3d",
    "errorbar_plot",
    "box_whisker_plot",
    "beeswarm_plot",
]
