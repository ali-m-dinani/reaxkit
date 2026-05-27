"""Renderer for ``beeswarm_plot``."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from reaxkit.presentation.plot.renderers.base import PlotRenderer, merged, save_or_show


class BeeswarmPlotRenderer(PlotRenderer):
    """Render seaborn beeswarm (swarmplot) with value-based color mapping."""

    def render(self, result, style=None):
        cfg = merged(result, style)
        x = cfg.get("x")
        y = cfg.get("y")
        hue = cfg.get("hue")
        if x is None or y is None:
            raise ValueError("beeswarm_plot requires 'x' and 'y'.")

        df = pd.DataFrame({"x": x, "y": y})
        if hue is None:
            hue = x
        df["hue"] = hue
        df = df.dropna(subset=["x", "y", "hue"])
        if df.empty:
            fig, ax = plt.subplots(figsize=(7.6, 3.4))
            ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
            ax.axis("off")
            return save_or_show(fig, cfg)

        fig = plt.figure(figsize=cfg.get("figsize", (9.2, max(3.4, 0.32 * df["y"].nunique()))))
        ax = fig.add_subplot(111)
        ax.axvline(0.0, c="grey", alpha=0.8, linewidth=1.0)
        marker_size = float(cfg.get("size", 3.5))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            ax = sns.swarmplot(
                data=df,
                x="x",
                y="y",
                hue="hue",
                palette=cfg.get("palette", "coolwarm"),
                size=marker_size,
                ax=ax,
            )
        placement_warning = any("cannot be placed" in str(w.message).lower() for w in caught)
        if placement_warning:
            ax.cla()
            ax.axvline(0.0, c="grey", alpha=0.8, linewidth=1.0)
            ax = sns.stripplot(
                data=df,
                x="x",
                y="y",
                hue="hue",
                palette=cfg.get("palette", "coolwarm"),
                size=max(2.5, marker_size - 0.8),
                jitter=float(cfg.get("jitter", 0.25)),
                alpha=float(cfg.get("alpha", 0.8)),
                ax=ax,
            )

        # Axis range: keep a readable near-1 window when sensitivity values cluster around 1.
        vals = pd.to_numeric(df["x"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size:
            xmin = float(np.nanmin(vals))
            xmax = float(np.nanmax(vals))
            p01 = float(np.nanpercentile(vals, 1))
            p99 = float(np.nanpercentile(vals, 99))
            median = float(np.nanmedian(vals))
            span = max(p99 - p01, xmax - xmin)
            if 0.95 <= median <= 1.05 and span <= 0.35:
                ax.set_xlim(0.9, 1.1)
            else:
                pad = max(0.02 * max(abs(p01), abs(p99), 1.0), 0.01)
                ax.set_xlim(p01 - pad, p99 + pad)

        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        lower_handle = Line2D([0], [0], marker="o", color="w", markerfacecolor="#3B4CC0", markersize=7, label="Lower parameter value")
        higher_handle = Line2D([0], [0], marker="o", color="w", markerfacecolor="#B40426", markersize=7, label="Higher parameter value")
        ax.legend(
            handles=[lower_handle, higher_handle],
            loc=str(cfg.get("legend_loc", "best")),
            frameon=False,
            title=str(cfg.get("legend_title", "Color Meaning")),
        )
        ax.spines["left"].set_visible(True)
        ax.grid(axis="x")
        ax.set_xlabel(cfg.get("xlabel", "Value"))
        ax.set_ylabel(cfg.get("ylabel", ""))
        title = cfg.get("title")
        if title:
            ax.set_title(str(title))
        fig.tight_layout()
        return save_or_show(fig, cfg)
