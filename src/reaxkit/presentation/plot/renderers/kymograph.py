"""Render a sampled field with a single color scale across frames."""

import matplotlib.pyplot as plt
import numpy as np

from reaxkit.presentation.plot.renderers.base import PlotRenderer, merged, save_or_show


def _bin_edges(centers):
    centers = np.asarray(centers, dtype=float)
    if len(centers) == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    midpoints = (centers[:-1] + centers[1:]) / 2
    return np.r_[centers[0] - (midpoints[0] - centers[0]), midpoints, centers[-1] + (centers[-1] - midpoints[-1])]


class KymographRenderer(PlotRenderer):
    def render(self, result, style=None):
        cfg = merged(result, style)
        figure, axes = plt.subplots(figsize=cfg.get("figsize", (9, 5)))
        mesh = axes.pcolormesh(
            _bin_edges(cfg["x"]), _bin_edges(cfg["y"]), np.asarray(cfg["z"]),
            shading="flat", cmap=cfg.get("cmap", "viridis"),
            vmin=cfg.get("vmin", 0.0), vmax=cfg.get("vmax"), rasterized=True,
        )
        axes.set(xlabel=cfg.get("xlabel", "Frame index"), ylabel=cfg.get("ylabel", "r (Å)"), title=cfg.get("title", "RDF evolution"))
        figure.colorbar(mesh, ax=axes, label=cfg.get("colorbar_label", "g(r)"))
        return save_or_show(figure, cfg)
