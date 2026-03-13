"""Renderer for box-whisker plots."""

from __future__ import annotations

import matplotlib.pyplot as plt

from reaxkit.presentation.plot.renderers.base import PlotRenderer, merged


class BoxWhiskerPlotRenderer(PlotRenderer):
    """Render matplotlib box-whisker plots."""

    def render(self, result, style=None):
        cfg = merged(result, style)
        data = cfg.get("data")
        labels = cfg.get("labels")
        title = cfg.get("title")
        xlabel = cfg.get("xlabel")
        ylabel = cfg.get("ylabel")
        save = cfg.get("save")
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
        if save:
            from pathlib import Path

            p = Path(save)
            exts = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".tif", ".tiff", ".bmp"}
            if p.suffix.lower() in exts:
                p.parent.mkdir(parents=True, exist_ok=True)
                out = p
            else:
                p.mkdir(parents=True, exist_ok=True)
                out = p / f"{(title or 'box_whisker_plot').replace(' ', '_')}.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
        return fig

