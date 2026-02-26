"""Renderer for ``dual_yaxis_plot``."""

from __future__ import annotations

import matplotlib.pyplot as plt

from reaxkit.presentation.plot.renderers.base import PlotRenderer, merged, save_or_show


class DualYaxisPlotRenderer(PlotRenderer):
    """Render dual y-axis plot."""

    def render(self, result, style=None):
        cfg = merged(result, style)
        x = cfg.get("x")
        y1 = cfg.get("y1")
        y2 = cfg.get("y2")
        fig, ax1 = plt.subplots(figsize=cfg.get("figsize", (10, 6)))
        ax1.plot(
            x,
            y1,
            linestyle=cfg.get("linestyle1", "-"),
            marker=cfg.get("marker1", ""),
            color=cfg.get("color1", "blue"),
        )
        ax1.set_xlabel(cfg.get("xlabel", ""))
        ax1.set_ylabel(cfg.get("ylabel1", ""), color=cfg.get("color1", "blue"))
        if cfg.get("xlim"):
            ax1.set_xlim(cfg["xlim"])
        if cfg.get("ylim1"):
            ax1.set_ylim(cfg["ylim1"])
        if cfg.get("grid", False):
            ax1.grid(True)
        if cfg.get("hline1") is not None:
            params = {"color": cfg.get("color1", "blue"), "linestyle": "--", "linewidth": 1}
            if cfg.get("hline1_kwargs"):
                params.update(cfg["hline1_kwargs"])
            ax1.axhline(cfg["hline1"], **params)

        ax2 = ax1.twinx()
        ax2.plot(
            x,
            y2,
            linestyle=cfg.get("linestyle2", "--"),
            marker=cfg.get("marker2", ""),
            color=cfg.get("color2", "green"),
        )
        ax2.set_ylabel(cfg.get("ylabel2", ""), color=cfg.get("color2", "green"))
        if cfg.get("ylim2"):
            ax2.set_ylim(cfg["ylim2"])
        if cfg.get("hline2") is not None:
            params = {"color": cfg.get("color2", "green"), "linestyle": "--", "linewidth": 1}
            if cfg.get("hline2_kwargs"):
                params.update(cfg["hline2_kwargs"])
            ax2.axhline(cfg["hline2"], **params)

        if cfg.get("vline") is not None:
            params = {"color": "black", "linestyle": ":", "linewidth": 1}
            if cfg.get("vline_kwargs"):
                params.update(cfg["vline_kwargs"])
            ax1.axvline(cfg["vline"], **params)
        if cfg.get("title"):
            fig.suptitle(cfg["title"])
        return save_or_show(fig, cfg)
