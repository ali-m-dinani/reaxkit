"""
Renderer for ``single_plot``.

**Usage context**

- Import these helpers from presentation workflows that produce tables, files, or plots.
- Reuse the public APIs here to keep output formatting and artifact behavior consistent.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt

from reaxkit.presentation.plot.renderers.base import PlotRenderer, merged


class SinglePlotRenderer(PlotRenderer):
    """Render line/scatter plot with optional multi-series."""

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
        from reaxkit.presentation.plot.renderers.single import SinglePlotRenderer
        instance = SinglePlotRenderer(...)
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
        series = cfg.get("series")
        hlines = cfg.get("hlines")
        title = cfg.get("title")
        xlabel = cfg.get("xlabel")
        ylabel = cfg.get("ylabel")
        save = cfg.get("save")
        legend = bool(cfg.get("legend", False))
        figsize = cfg.get("figsize", (8.0, 3.2))
        plot_type = cfg.get("plot_type_style") or cfg.get("kind") or cfg.get("series_type", "line")

        fig, ax = plt.subplots(figsize=figsize)

        def _plot(ax_, sx, sy, label=None, **kwargs):
            if plot_type == "scatter":
                ax_.scatter(sx, sy, label=label, **kwargs)
            else:
                ax_.plot(sx, sy, label=label, **kwargs)

        if series is not None:
            for s in series:
                sx = s.get("x")
                sy = s.get("y")
                if sx is None or sy is None:
                    continue
                lbl = s.get("label")
                lw = s.get("linewidth", 1.2)
                mk = s.get("marker", "." if plot_type == "scatter" else None)
                ms = s.get("markersize", 4)
                al = s.get("alpha", 1.0)
                kwargs = dict(linewidth=lw, marker=mk, alpha=al)
                if plot_type == "scatter":
                    kwargs["s"] = ms
                else:
                    kwargs["markersize"] = ms
                _plot(ax, sx, sy, label=lbl, **kwargs)
        else:
            if x is None or y is None:
                raise ValueError("Provide (x, y) or 'series=[...]'.")
            if plot_type == "scatter":
                ax.scatter(x, y, label=None)
            else:
                ax.plot(x, y, label=None)

        if hlines:
            for h in hlines:
                if isinstance(h, Mapping):
                    yv = h.get("y")
                    if yv is None:
                        continue
                    lbl = h.get("label")
                    ls = h.get("linestyle", "--")
                    lw = h.get("linewidth", 1.0)
                    al = h.get("alpha", 1.0)
                    ax.axhline(yv, linestyle=ls, linewidth=lw, alpha=al, color="gray", label=lbl)
                elif isinstance(h, tuple):
                    yv, lbl = h[0], (h[1] if len(h) > 1 else None)
                    ax.axhline(yv, linestyle="--", linewidth=1.0, alpha=1.0, color="gray", label=lbl)
                else:
                    ax.axhline(float(h), linestyle="--", linewidth=1.0, alpha=1.0, color="gray")

        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
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
                out = p / f"{(title or 'single_plot').replace(' ', '_')}.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
        return fig
