"""Renderer for ``heatmap2d_from_3d``."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from reaxkit.presentation.plot.renderers.base import PlotRenderer, merged, save_or_show


class Heatmap2DRenderer(PlotRenderer):
    """Render 2D heatmap projected from 3D points."""

    def render(self, result, style=None):
        cfg = merged(result, style)
        coords = np.asarray(cfg.get("coords"), float)
        values = np.asarray(cfg.get("values"), float)
        assert coords.ndim == 2 and coords.shape[1] == 3, "coords must be (N,3)"
        assert values.ndim == 1 and len(values) == len(coords), "values must be (N,)"

        plane = cfg.get("plane", "xy")
        if plane == "xy":
            u, v = coords[:, 0], coords[:, 1]
            xlabel, ylabel = "x (A)", "y (A)"
        elif plane == "xz":
            u, v = coords[:, 0], coords[:, 2]
            xlabel, ylabel = "x (A)", "z (A)"
        elif plane == "yz":
            u, v = coords[:, 1], coords[:, 2]
            xlabel, ylabel = "y (A)", "z (A)"
        else:
            raise ValueError("plane must be one of {'xy','xz','yz'}")

        bins = cfg.get("bins", 50)
        if isinstance(bins, int):
            nx = ny = int(bins)
        else:
            nx, ny = int(bins[0]), int(bins[1])

        xlim = cfg.get("xlim")
        ylim = cfg.get("ylim")
        umin = np.min(u) if xlim is None else xlim[0]
        umax = np.max(u) if xlim is None else xlim[1]
        vmin_edge = np.min(v) if ylim is None else ylim[0]
        vmax_edge = np.max(v) if ylim is None else ylim[1]

        xedges = np.linspace(umin, umax, nx + 1)
        yedges = np.linspace(vmin_edge, vmax_edge, ny + 1)
        ui = np.clip(np.digitize(u, xedges) - 1, 0, nx - 1)
        vi = np.clip(np.digitize(v, yedges) - 1, 0, ny - 1)
        flat_idx = vi * nx + ui
        n_cells = nx * ny
        grid = np.full((ny, nx), np.nan, float)

        agg = cfg.get("agg", "mean")
        if isinstance(agg, str):
            agg_lower = agg.lower()
            if agg_lower == "count":
                cnt = np.bincount(flat_idx, minlength=n_cells).astype(float)
                grid = cnt.reshape(ny, nx)
            elif agg_lower in {"sum", "mean"}:
                sumv = np.bincount(flat_idx, weights=values, minlength=n_cells).astype(float)
                cnt = np.bincount(flat_idx, minlength=n_cells).astype(float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    if agg_lower == "sum":
                        grid = sumv.reshape(ny, nx)
                    else:
                        grid = (sumv / cnt).reshape(ny, nx)
                        grid[cnt.reshape(ny, nx) == 0] = np.nan
            elif agg_lower in {"max", "min"}:
                fill_val = -np.inf if agg_lower == "max" else np.inf
                flat_grid = np.full(n_cells, fill_val, float)
                for idx, val in zip(flat_idx, values):
                    if agg_lower == "max":
                        if val > flat_grid[idx]:
                            flat_grid[idx] = val
                    else:
                        if val < flat_grid[idx]:
                            flat_grid[idx] = val
                if agg_lower == "max":
                    flat_grid[flat_grid == -np.inf] = np.nan
                else:
                    flat_grid[flat_grid == np.inf] = np.nan
                grid = flat_grid.reshape(ny, nx)
            else:
                raise ValueError("agg must be one of {'mean','max','min','sum','count'} or callable")
        else:
            buckets = [list() for _ in range(n_cells)]
            for idx, val in zip(flat_idx, values):
                buckets[idx].append(val)
            flat_grid = np.full(n_cells, np.nan, float)
            for i, bucket in enumerate(buckets):
                if bucket:
                    try:
                        flat_grid[i] = agg(np.asarray(bucket, float))
                    except Exception:
                        flat_grid[i] = np.nan
            grid = flat_grid.reshape(ny, nx)

        cmin = np.nanmin(grid) if cfg.get("vmin") is None else cfg.get("vmin")
        cmax = np.nanmax(grid) if cfg.get("vmax") is None else cfg.get("vmax")
        fig, ax = plt.subplots(figsize=cfg.get("figsize", (6.5, 5.5)))
        extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
        im = ax.imshow(
            grid,
            origin="lower",
            extent=extent,
            aspect="auto",
            vmin=cmin,
            vmax=cmax,
            cmap=cfg.get("cmap", "viridis"),
            interpolation="nearest",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(cfg.get("title", "2D aggregated heatmap"))
        if bool(cfg.get("show_colorbar", True)):
            cb = fig.colorbar(im, ax=ax, pad=0.02)
            cb.set_label(f"{agg if isinstance(agg, str) else 'agg'} of values")
        return save_or_show(fig, cfg)
