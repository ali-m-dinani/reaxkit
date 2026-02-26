"""Base interface and shared helpers for plot renderers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Optional

import matplotlib.pyplot as plt


class PlotRenderer(ABC):
    """Abstract renderer interface."""

    @abstractmethod
    def render(self, result, style=None):
        """Render a plot from a result payload and optional style."""


def as_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return dict(result)
    if hasattr(result, "__dict__"):
        return dict(result.__dict__)
    raise TypeError("result must be dict-like or have __dict__ attributes")


def merged(result: Any, style: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    data = as_dict(result)
    if style:
        data.update(dict(style))
    return data


def save_or_show(fig: plt.Figure, cfg: Mapping[str, Any]) -> plt.Figure:
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

