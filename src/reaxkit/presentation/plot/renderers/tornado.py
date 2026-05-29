"""
Renderer for ``tornado_plot``.

**Usage context**

- Import these helpers from presentation workflows that produce tables, files, or plots.
- Reuse the public APIs here to keep output formatting and artifact behavior consistent.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reaxkit.presentation.plot.renderers.base import PlotRenderer, merged, save_or_show


class TornadoPlotRenderer(PlotRenderer):
    """Render tornado sensitivity plot."""

    def render(self, result, style=None):
        """
        Render.
        
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
        from reaxkit.presentation.plot.renderers.tornado import TornadoPlotRenderer
        instance = TornadoPlotRenderer(...)
        result = instance.render(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        cfg = merged(result, style)
        labels = cfg.get("labels")
        min_vals = cfg.get("min_vals")
        max_vals = cfg.get("max_vals")
        median_vals = cfg.get("median_vals")
        df = pd.DataFrame({"label": labels, "min": min_vals, "max": max_vals})
        if median_vals is not None:
            if len(median_vals) != len(labels):
                raise ValueError("median_vals must be same length as labels/min_vals/max_vals")
            df["median"] = list(median_vals)
        df["span"] = df["max"] - df["min"]
        df = df.sort_values("span", ascending=False).reset_index(drop=True)
        top = int(cfg.get("top", 0))
        if top > 0:
            df = df.head(top)
        if df.empty:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
            ax.axis("off")
            return save_or_show(fig, cfg)

        fig_height = max(3.2, 0.38 * len(df))
        fig, ax = plt.subplots(figsize=(8.6, fig_height))
        y_positions = np.arange(len(df))
        bar_height = 0.5
        edge_common = dict(edgecolor="black", linewidth=0.5)
        vline = cfg.get("vline")
        left_color = cfg.get("left_color", "#1F77B4")
        right_color = cfg.get("right_color", (225 / 255, 113 / 255, 29 / 255))
        for y, row in zip(y_positions, df.itertuples(index=False)):
            left, right = (row.max, row.min) if row.max < row.min else (row.min, row.max)
            if vline is None:
                ax.barh(y=y, width=right - left, left=left, height=bar_height, color="tab:gray", alpha=0.7, **edge_common)
            else:
                if right <= vline:
                    ax.barh(y=y, width=right - left, left=left, height=bar_height, color=left_color, alpha=0.8, **edge_common)
                elif left >= vline:
                    ax.barh(y=y, width=right - left, left=left, height=bar_height, color=right_color, alpha=0.8, **edge_common)
                else:
                    ax.barh(y=y, width=vline - left, left=left, height=bar_height, color=left_color, alpha=0.8, **edge_common)
                    ax.barh(y=y, width=right - vline, left=vline, height=bar_height, color=right_color, alpha=0.8, **edge_common)
            if "median" in df.columns and not pd.isna(row.median):
                ax.plot(row.median, y, marker="*", markersize=5, color="black", zorder=5)
        if vline is not None:
            ax.axvline(vline, linestyle="--", linewidth=1, color=(66 / 255, 196 / 255, 127 / 255))
        ax.set_yticks(list(y_positions))
        ax.set_yticklabels(df["label"])
        ax.invert_yaxis()
        ax.set_xlabel(cfg.get("xlabel", "Value"))
        ax.set_ylabel(cfg.get("ylabel", "Value"))
        ax.set_title(cfg.get("title", "Tornado Plot"))
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        fig.tight_layout()
        return save_or_show(fig, cfg)
