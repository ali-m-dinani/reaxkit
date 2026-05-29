"""
Renderer for error-bar plots.

**Usage context**

- Import these helpers from presentation workflows that produce tables, files, or plots.
- Reuse the public APIs here to keep output formatting and artifact behavior consistent.
"""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt

from reaxkit.presentation.plot.renderers.base import PlotRenderer, merged


class ErrorbarPlotRenderer(PlotRenderer):
    """Render matplotlib error-bar plots."""

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
        from reaxkit.presentation.plot.renderers.errorbar import ErrorbarPlotRenderer
        instance = ErrorbarPlotRenderer(...)
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
        x = cfg.get("x")
        y = cfg.get("y")
        yerr = cfg.get("yerr")
        xerr = cfg.get("xerr")
        series = cfg.get("series")
        title = cfg.get("title")
        xlabel = cfg.get("xlabel")
        ylabel = cfg.get("ylabel")
        save = cfg.get("save")
        legend = bool(cfg.get("legend", False))
        figsize = cfg.get("figsize", (8.0, 4.0))
        fmt = cfg.get("fmt", "o-")
        capsize = cfg.get("capsize", 4)
        alpha = float(cfg.get("alpha", 1.0))

        fig, ax = plt.subplots(figsize=figsize)

        if series is not None:
            for item in series:
                if not isinstance(item, Mapping):
                    continue
                sx = item.get("x")
                sy = item.get("y")
                if sx is None or sy is None:
                    continue
                ax.errorbar(
                    sx,
                    sy,
                    yerr=item.get("yerr"),
                    xerr=item.get("xerr"),
                    fmt=item.get("fmt", fmt),
                    capsize=item.get("capsize", capsize),
                    alpha=float(item.get("alpha", alpha)),
                    label=item.get("label"),
                )
        else:
            if x is None or y is None:
                raise ValueError("Provide (x, y) or 'series=[...]' for errorbar_plot.")
            ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=fmt, capsize=capsize, alpha=alpha)

        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if bool(cfg.get("grid", True)):
            ax.grid(True, alpha=0.3)
        if legend:
            ax.legend()

        fig.tight_layout()
        if save:
            from pathlib import Path

            p = Path(save)
            exts = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".tif", ".tiff", ".bmp"}
            if p.suffix.lower() in exts:
                p.parent.mkdir(parents=True, exist_ok=True)
                out = p
            else:
                p.mkdir(parents=True, exist_ok=True)
                out = p / f"{(title or 'errorbar_plot').replace(' ', '_')}.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
        return fig

