"""Renderer for ``directed_plot``."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from reaxkit.presentation.plot.renderers.base import PlotRenderer, merged, save_or_show


class DirectedPlotRenderer(PlotRenderer):
    """Render a directed 2D path."""

    def render(self, result, style=None):
        cfg = merged(result, style)
        x = np.asarray(cfg.get("x"), dtype=float)
        y = np.asarray(cfg.get("y"), dtype=float)
        dx = np.diff(x)
        dy = np.diff(y)

        fig, ax = plt.subplots(figsize=cfg.get("figsize", (10, 6)))
        ax.plot(x, y, linestyle=cfg.get("linestyle", "-"), color=cfg.get("color", "blue"), label="Path")
        ax.quiver(
            x[:-1],
            y[:-1],
            dx,
            dy,
            angles="xy",
            scale_units="xy",
            scale=1,
            color=cfg.get("arrow_color", "red"),
            width=float(cfg.get("arrow_width", 0.003)),
        )
        ax.set(
            title=cfg.get("title", ""),
            xlabel=cfg.get("xlabel", ""),
            ylabel=cfg.get("ylabel", ""),
        )
        if cfg.get("xlim"):
            ax.set_xlim(cfg["xlim"])
        if cfg.get("ylim"):
            ax.set_ylim(cfg["ylim"])
        if cfg.get("grid", False):
            ax.grid(True)
        if cfg.get("hline") is not None:
            params = {"color": "black", "linestyle": "--", "linewidth": 1}
            if cfg.get("hline_kwargs"):
                params.update(cfg["hline_kwargs"])
            ax.axhline(cfg["hline"], **params)
        if cfg.get("legend", False):
            ax.legend()
        return save_or_show(fig, cfg)
