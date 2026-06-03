"""
Base interface and shared helpers for plot renderers.

**Usage context**

- Import these helpers from presentation workflows that produce tables, files, or plots.
- Reuse the public APIs here to keep output formatting and artifact behavior consistent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Optional

import matplotlib.pyplot as plt


class PlotRenderer(ABC):
    """Abstract renderer interface."""

    @abstractmethod
    def render(self, result, style=None):
        """
        Render a plot from a result payload and optional style.
        
        This function is part of the ReaxKit presentation API and performs the operation
        described by its name and arguments.
        
        Parameters
        -----
        result : Any
            Input parameter used by this function.
        style : Any, optional
            Input parameter used by this function.
        
        Returns
        -----
        Any
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.presentation.plot.renderers.base import PlotRenderer
        instance = PlotRenderer(...)
        result = instance.render(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """


def as_dict(result: Any) -> dict[str, Any]:
    """
    As dict.
    
    This function is part of the ReaxKit presentation API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    result : Any
        Input parameter used by this function.
    
    Returns
    -----
    dict[str, Any]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.presentation.plot.renderers.base import as_dict
    result = as_dict(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    if isinstance(result, dict):
        return dict(result)
    if hasattr(result, "__dict__"):
        return dict(result.__dict__)
    raise TypeError("result must be dict-like or have __dict__ attributes")


def merged(result: Any, style: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """
    Merged.
    
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
    dict[str, Any]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.presentation.plot.renderers.base import merged
    result = merged(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    data = as_dict(result)
    if style:
        data.update(dict(style))
    return data


def save_or_show(fig: plt.Figure, cfg: Mapping[str, Any]) -> plt.Figure:
    """
    Save or show.
    
    This function is part of the ReaxKit presentation API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    fig : plt.Figure
        Input parameter used by this function.
    cfg : Mapping[str, Any]
        Input parameter used by this function.
    
    Returns
    -----
    plt.Figure
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.presentation.plot.renderers.base import save_or_show
    result = save_or_show(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    save = cfg.get("save")
    title = str(cfg.get("title") or cfg.get("plot_type") or "plot")
    if save:
        p = Path(save)
        exts = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".tif", ".tiff", ".bmp"}
        if p.suffix.lower() in exts:
            p.parent.mkdir(parents=True, exist_ok=True)
            out = p
        else:
            p.mkdir(parents=True, exist_ok=True)
            out = p / f"{title.replace(' ', '_')}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig
