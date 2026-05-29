"""
Renderer-based plotting API.

**Usage context**

- Import these helpers from presentation workflows that produce tables, files, or plots.
- Reuse the public APIs here to keep output formatting and artifact behavior consistent.
"""

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
    """
    Single plot.
    
    This function is part of the ReaxKit presentation API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    x : Any, optional
        Input parameter used by this function.
    y : Any, optional
        Input parameter used by this function.
    **kwargs : Any
        Input parameter used by this function.
    
    Returns
    -----
    Any
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.presentation.plot.__init__ import single_plot
    result = single_plot(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return plot({"plot_type": "single_plot", "x": x, "y": y, **kwargs})


def directed_plot(x, y, **kwargs):
    """
    Directed plot.
    
    This function is part of the ReaxKit presentation API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    x : Any
        Input parameter used by this function.
    y : Any
        Input parameter used by this function.
    **kwargs : Any
        Input parameter used by this function.
    
    Returns
    -----
    Any
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.presentation.plot.__init__ import directed_plot
    result = directed_plot(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return plot({"plot_type": "directed_plot", "x": x, "y": y, **kwargs})


def dual_yaxis_plot(x, y1, y2, **kwargs):
    """
    Dual yaxis plot.
    
    This function is part of the ReaxKit presentation API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    x : Any
        Input parameter used by this function.
    y1 : Any
        Input parameter used by this function.
    y2 : Any
        Input parameter used by this function.
    **kwargs : Any
        Input parameter used by this function.
    
    Returns
    -----
    Any
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.presentation.plot.__init__ import dual_yaxis_plot
    result = dual_yaxis_plot(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return plot({"plot_type": "dual_yaxis_plot", "x": x, "y1": y1, "y2": y2, **kwargs})


def multi_subplots(subplots, **kwargs):
    """
    Multi subplots.
    
    This function is part of the ReaxKit presentation API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    subplots : Any
        Input parameter used by this function.
    **kwargs : Any
        Input parameter used by this function.
    
    Returns
    -----
    Any
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.presentation.plot.__init__ import multi_subplots
    result = multi_subplots(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return plot({"plot_type": "multi_subplots", "subplots": subplots, **kwargs})


def tornado_plot(labels, min_vals, max_vals, **kwargs):
    """
    Tornado plot.
    
    This function is part of the ReaxKit presentation API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    labels : Any
        Input parameter used by this function.
    min_vals : Any
        Input parameter used by this function.
    max_vals : Any
        Input parameter used by this function.
    **kwargs : Any
        Input parameter used by this function.
    
    Returns
    -----
    Any
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.presentation.plot.__init__ import tornado_plot
    result = tornado_plot(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return plot({"plot_type": "tornado_plot", "labels": labels, "min_vals": min_vals, "max_vals": max_vals, **kwargs})


def scatter3d_points(coords, values, **kwargs):
    """
    Scatter3d points.
    
    This function is part of the ReaxKit presentation API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    coords : Any
        Input parameter used by this function.
    values : Any
        Input parameter used by this function.
    **kwargs : Any
        Input parameter used by this function.
    
    Returns
    -----
    Any
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.presentation.plot.__init__ import scatter3d_points
    result = scatter3d_points(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return plot({"plot_type": "scatter3d_points", "coords": coords, "values": values, **kwargs})


def heatmap2d_from_3d(coords, values, **kwargs):
    """
    Heatmap2d from 3d.
    
    This function is part of the ReaxKit presentation API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    coords : Any
        Input parameter used by this function.
    values : Any
        Input parameter used by this function.
    **kwargs : Any
        Input parameter used by this function.
    
    Returns
    -----
    Any
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.presentation.plot.__init__ import heatmap2d_from_3d
    result = heatmap2d_from_3d(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return plot({"plot_type": "heatmap2d_from_3d", "coords": coords, "values": values, **kwargs})


def errorbar_plot(x=None, y=None, yerr=None, **kwargs):
    """
    Errorbar plot.
    
    This function is part of the ReaxKit presentation API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    x : Any, optional
        Input parameter used by this function.
    y : Any, optional
        Input parameter used by this function.
    yerr : Any, optional
        Input parameter used by this function.
    **kwargs : Any
        Input parameter used by this function.
    
    Returns
    -----
    Any
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.presentation.plot.__init__ import errorbar_plot
    result = errorbar_plot(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return plot({"plot_type": "errorbar_plot", "x": x, "y": y, "yerr": yerr, **kwargs})


def box_whisker_plot(data, labels=None, **kwargs):
    """
    Box whisker plot.
    
    This function is part of the ReaxKit presentation API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    data : Any
        Input parameter used by this function.
    labels : Any, optional
        Input parameter used by this function.
    **kwargs : Any
        Input parameter used by this function.
    
    Returns
    -----
    Any
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.presentation.plot.__init__ import box_whisker_plot
    result = box_whisker_plot(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return plot({"plot_type": "box_whisker_plot", "data": data, "labels": labels, **kwargs})


def beeswarm_plot(x, y, **kwargs):
    """
    Beeswarm plot.
    
    This function is part of the ReaxKit presentation API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    x : Any
        Input parameter used by this function.
    y : Any
        Input parameter used by this function.
    **kwargs : Any
        Input parameter used by this function.
    
    Returns
    -----
    Any
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.presentation.plot.__init__ import beeswarm_plot
    result = beeswarm_plot(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
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
