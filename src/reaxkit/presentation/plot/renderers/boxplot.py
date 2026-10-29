"""
Renderer for box-whisker plots.

**Usage context**

- Import these helpers from presentation workflows that produce tables, files, or plots.
- Reuse the public APIs here to keep output formatting and artifact behavior consistent.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from reaxkit.presentation.plot.renderers.base import PlotRenderer, merged, save_or_show


class BoxWhiskerPlotRenderer(PlotRenderer):
    """Render matplotlib box-whisker plots."""

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
        from reaxkit.presentation.plot.renderers.boxplot import BoxWhiskerPlotRenderer
        instance = BoxWhiskerPlotRenderer(...)
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
        data = cfg.get("data")
        labels = cfg.get("labels")
        title = cfg.get("title")
        xlabel = cfg.get("xlabel")
        ylabel = cfg.get("ylabel")
        figsize = cfg.get("figsize", (8.0, 4.5))
        notch = bool(cfg.get("notch", False))
        showfliers = bool(cfg.get("showfliers", True))
        patch_artist = bool(cfg.get("patch_artist", True))

        if data is None:
            raise ValueError("box_whisker_plot requires 'data' as a list of numeric series.")

        fig, ax = plt.subplots(figsize=figsize)
        bp = ax.boxplot(
            data,
            labels=labels,
            notch=notch,
            showfliers=showfliers,
            patch_artist=patch_artist,
        )

        if patch_artist:
            colors = cfg.get("box_colors")
            if isinstance(colors, list) and colors:
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)

        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if bool(cfg.get("grid", True)):
            ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        return save_or_show(fig, {"plot_type": "box_whisker_plot", **cfg})

