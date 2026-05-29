"""
Renderer for ``wireframe3d_subplots``.

**Usage context**

- Import these helpers from presentation workflows that produce tables, files, or plots.
- Reuse the public APIs here to keep output formatting and artifact behavior consistent.
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np

from reaxkit.presentation.plot.renderers.base import PlotRenderer, merged, save_or_show


class Wireframe3DSubplotsRenderer(PlotRenderer):
    """Render multiple 3D wireframe subplot panels."""

    @staticmethod
    def _grid_shape(grid, nplots: int) -> tuple[int, int]:
        """
        Grid shape.
        """
        if not grid:
            return (nplots, 1)
        if isinstance(grid, (tuple, list)) and len(grid) == 2:
            rows, cols = int(grid[0]), int(grid[1])
        else:
            match = re.fullmatch(r"\s*(\d+)\s*[xX*]\s*(\d+)\s*", str(grid))
            if not match:
                raise ValueError("grid must look like '2x2' or '2*2'")
            rows, cols = int(match.group(1)), int(match.group(2))
        if rows <= 0 or cols <= 0:
            raise ValueError("grid rows and columns must be positive")
        return rows, cols

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
        from reaxkit.presentation.plot.renderers.wireframe3d_subplots import Wireframe3DSubplotsRenderer
        instance = Wireframe3DSubplotsRenderer(...)
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
        subplots = cfg.get("subplots") or []
        nplots = len(subplots)

        if nplots == 0:
            fig = plt.figure(figsize=cfg.get("figsize", (7.5, 6.0)))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
            ax.axis("off")
            return save_or_show(fig, cfg)

        rows, cols = self._grid_shape(cfg.get("grid"), nplots)
        fig = plt.figure(figsize=cfg.get("figsize", (4.8 * cols, 4.6 * rows)))

        for i, item in enumerate(subplots, start=1):
            ax = fig.add_subplot(rows, cols, i, projection="3d")
            segments = item.get("segments") or []
            points = np.asarray(item.get("points") or [], dtype=float)
            values = np.asarray(item.get("values") or [], dtype=float)

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
                    linewidth=float(cfg.get("line_width", 0.7)),
                    alpha=float(cfg.get("line_alpha", 0.55)),
                )

            if points.ndim == 2 and points.shape[1] == 3 and len(points):
                scatter_kwargs = {
                    "s": float(cfg.get("point_size", 7.0)),
                    "alpha": float(cfg.get("point_alpha", 0.8)),
                    "depthshade": True,
                }
                if values.ndim == 1 and len(values) == len(points):
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=values, cmap=cfg.get("cmap", "viridis"), **scatter_kwargs)
                else:
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=cfg.get("point_color", "tab:blue"), **scatter_kwargs)

            ax.set_title(str(item.get("title", f"subplot {i}")), fontsize=10)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=float(cfg.get("elev", 22.0)), azim=float(cfg.get("azim", 38.0)))

        for j in range(nplots + 1, rows * cols + 1):
            ax = fig.add_subplot(rows, cols, j)
            ax.axis("off")

        title = cfg.get("title")
        if title:
            fig.suptitle(str(title), fontsize=14, y=0.98)
        fig.tight_layout()
        return save_or_show(fig, cfg)

