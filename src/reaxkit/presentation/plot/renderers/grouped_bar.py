"""Renderer for grouped categorical bar plots."""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np

from reaxkit.presentation.plot.renderers.base import PlotRenderer, merged, save_or_show


class GroupedBarPlotRenderer(PlotRenderer):
    """Render multiple consistently colored bars for each categorical label."""

    def render(self, result, style=None):
        cfg = merged(result, style)
        labels = [str(value) for value in (cfg.get("labels") or [])]
        series = list(cfg.get("series") or [])
        if not labels or not series:
            raise ValueError("grouped_bar_plot requires non-empty 'labels' and 'series'.")

        count = len(labels)
        x = np.arange(count, dtype=float)
        width = float(cfg.get("group_width", 0.8)) / max(1, len(series))
        figsize = cfg.get("figsize", (max(8.0, count * 2.2), 5.0))
        fig, ax = plt.subplots(figsize=figsize)

        for index, item in enumerate(series):
            values = list(item.get("values") or [])
            if len(values) != count:
                raise ValueError("Each grouped-bar series must have one value per label.")
            offset = (index - (len(series) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                values,
                width,
                label=item.get("label"),
                color=item.get("color"),
                alpha=float(item.get("alpha", 0.9)),
            )

        minimum_slots = max(
            count, int(cfg.get("minimum_category_slots", count))
        )
        if minimum_slots > count:
            empty_slots = minimum_slots - count
            left_padding = empty_slots / 2.0
            ax.set_xlim(-0.5 - left_padding, count - 0.5 + left_padding)

        wrap_width = int(cfg.get("label_wrap", 28))
        shown_labels = [textwrap.fill(label, width=wrap_width) for label in labels]
        ax.set_xticks(x)
        ax.set_xticklabels(
            shown_labels,
            rotation=float(cfg.get("label_rotation", 15)),
            ha=str(cfg.get("label_horizontal_alignment", "right")),
        )
        if cfg.get("title"):
            ax.set_title(str(cfg["title"]))
        if cfg.get("xlabel"):
            ax.set_xlabel(str(cfg["xlabel"]))
        if cfg.get("ylabel"):
            ax.set_ylabel(str(cfg["ylabel"]))
        if bool(cfg.get("zero_line", True)):
            ax.axhline(0.0, color="black", linewidth=0.8)
        if bool(cfg.get("grid", True)):
            ax.grid(True, axis="y", alpha=0.25)
        if bool(cfg.get("legend", True)):
            ax.legend()

        fig.tight_layout()
        return save_or_show(fig, cfg)


__all__ = ["GroupedBarPlotRenderer"]
