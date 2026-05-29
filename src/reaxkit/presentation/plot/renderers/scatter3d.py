"""
Renderer for ``scatter3d_points``.

**Usage context**

- Import these helpers from presentation workflows that produce tables, files, or plots.
- Reuse the public APIs here to keep output formatting and artifact behavior consistent.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from reaxkit.presentation.plot.renderers.base import PlotRenderer, merged, save_or_show


class Scatter3DRenderer(PlotRenderer):
    """Render 3D scatter points."""

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
        from reaxkit.presentation.plot.renderers.scatter3d import Scatter3DRenderer
        instance = Scatter3DRenderer(...)
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
        coords = np.asarray(cfg.get("coords"), float)
        values = np.asarray(cfg.get("values"), float)
        assert coords.ndim == 2 and coords.shape[1] == 3, "coords must be (N,3)"
        assert values.ndim == 1 and len(values) == len(coords), "values must be (N,)"

        fig = plt.figure(figsize=cfg.get("figsize", (7.5, 6.0)))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=values,
            cmap=cfg.get("cmap", "coolwarm"),
            vmin=cfg.get("vmin"),
            vmax=cfg.get("vmax"),
            s=float(cfg.get("s", 8.0)),
            alpha=float(cfg.get("alpha", 0.9)),
            depthshade=True,
        )
        if bool(cfg.get("show_colorbar", True)):
            cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
            cb.set_label("value")
        ax.set_xlabel("x (A)")
        ax.set_ylabel("y (A)")
        ax.set_zlabel("z (A)")
        ax.set_title(cfg.get("title", "atoms (3D)"))
        ax.view_init(elev=float(cfg.get("elev", 22.0)), azim=float(cfg.get("azim", 38.0)))
        return save_or_show(fig, cfg)
