"""Renderer for ``wireframe3d_plot``."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from reaxkit.presentation.plot.renderers.base import PlotRenderer, merged, save_or_show


class Wireframe3DRenderer(PlotRenderer):
    """Render 3D wireframe segments with optional site points."""

    def render(self, result, style=None):
        cfg = merged(result, style)
        segments = cfg.get("segments") or []
        points = np.asarray(cfg.get("points") or [], dtype=float)
        values = np.asarray(cfg.get("values") or [], dtype=float)

        fig = plt.figure(figsize=cfg.get("figsize", (7.5, 6.0)))
        ax = fig.add_subplot(111, projection="3d")

        for seg in segments:
            if not isinstance(seg, (list, tuple)) or len(seg) != 2:
                continue
            p0 = np.asarray(seg[0], dtype=float)
            p1 = np.asarray(seg[1], dtype=float)
            if p0.shape != (3,) or p1.shape != (3,):
                continue
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color=cfg.get("line_color", "black"),
                linewidth=float(cfg.get("line_width", 0.8)),
                alpha=float(cfg.get("line_alpha", 0.6)),
            )

        if points.ndim == 2 and points.shape[1] == 3 and len(points):
            scatter_kwargs = {
                "s": float(cfg.get("point_size", 8.0)),
                "alpha": float(cfg.get("point_alpha", 0.85)),
                "depthshade": True,
            }
            if values.ndim == 1 and len(values) == len(points):
                sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=values, cmap=cfg.get("cmap", "viridis"), **scatter_kwargs)
                if bool(cfg.get("show_colorbar", True)):
                    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
                    cb.set_label(str(cfg.get("colorbar_label", "value")))
            else:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=cfg.get("point_color", "tab:blue"), **scatter_kwargs)

        ax.set_xlabel("x (A)")
        ax.set_ylabel("y (A)")
        ax.set_zlabel("z (A)")
        ax.set_title(str(cfg.get("title", "Voronoi Diagram (3D)")))
        ax.view_init(elev=float(cfg.get("elev", 22.0)), azim=float(cfg.get("azim", 38.0)))
        return save_or_show(fig, cfg)

