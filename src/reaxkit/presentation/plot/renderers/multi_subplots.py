"""Renderer for ``multi_subplots``."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt

from reaxkit.presentation.plot.renderers.base import PlotRenderer, merged, save_or_show


class MultiSubplotsRenderer(PlotRenderer):
    """Render stacked subplots."""

    def render(self, result, style=None):
        cfg = merged(result, style)
        subplots = cfg.get("subplots")
        nplots = len(subplots)
        if nplots == 0:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
            ax.axis("off")
            return save_or_show(fig, cfg)

        def _normalize_seq(val, n: int):
            if val is None:
                return [None] * n
            if isinstance(val, (list, tuple)):
                if len(val) == 1:
                    return [val[0]] * n
                if len(val) != n:
                    raise ValueError(f"Expected sequence of length 1 or {n}, got {len(val)}")
                return list(val)
            return [val] * n

        title = cfg.get("title")
        if isinstance(title, (list, tuple)):
            per_titles = _normalize_seq(title, nplots)
            global_title = None
        else:
            per_titles = [None] * nplots
            global_title = title
        xlabels = _normalize_seq(cfg.get("xlabel"), nplots)
        ylabels = _normalize_seq(cfg.get("ylabel"), nplots)

        fig, axes = plt.subplots(
            nplots,
            1,
            figsize=cfg.get("figsize", (8.0, 6.0)),
            sharex=bool(cfg.get("sharex", False)),
            sharey=bool(cfg.get("sharey", False)),
            squeeze=False,
        )
        axes = axes.flatten()
        for idx, ax in enumerate(axes):
            for series in subplots[idx]:
                x = series.get("x")
                y = series.get("y")
                if x is None or y is None:
                    continue
                ax.plot(x, y, label=series.get("label"))
            if ylabels[idx]:
                ax.set_ylabel(ylabels[idx])
            if xlabels[idx]:
                ax.set_xlabel(xlabels[idx])
            if per_titles[idx]:
                ax.set_title(per_titles[idx])
            if cfg.get("grid", False):
                ax.grid(True, alpha=0.3)
            if cfg.get("legend", True):
                ax.legend(fontsize=9)
        if global_title:
            fig.suptitle(global_title, fontsize=14, y=0.98)
        fig.tight_layout()
        return save_or_show(fig, cfg)
