"""
Renderer for ``multi_subplots``.

**Usage context**

- Import these helpers from presentation workflows that produce tables, files, or plots.
- Reuse the public APIs here to keep output formatting and artifact behavior consistent.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt

from reaxkit.presentation.plot.renderers.base import PlotRenderer, merged, save_or_show


class MultiSubplotsRenderer(PlotRenderer):
    """Render stacked subplots."""

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

    @staticmethod
    def _paged_save_target(save, page_index: int, total_pages: int):
        """
        Paged save target.
        """
        if not save or total_pages == 1:
            return save
        p = Path(save)
        if p.suffix:
            return str(p.with_name(f"{p.stem}_{page_index + 1}{p.suffix}"))
        return str(p / f"page_{page_index + 1}")

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
        from reaxkit.presentation.plot.renderers.multi_subplots import MultiSubplotsRenderer
        instance = MultiSubplotsRenderer(...)
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
        subplots = cfg.get("subplots")
        nplots = len(subplots)
        if nplots == 0:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
            ax.axis("off")
            return save_or_show(fig, cfg)

        def _normalize_seq(val, n: int):
            if val is None:
                return [None] * n
            if isinstance(val, (list, tuple)):
                if len(val) == 1:
                    return [val[0]] * n
                if len(val) != n:
                    raise ValueError(f"Expected sequence of length 1 or {n}, got {len(val)}")
                return list(val)
            return [val] * n

        title = cfg.get("title")
        if isinstance(title, (list, tuple)):
            per_titles = _normalize_seq(title, nplots)
            global_title = None
        else:
            per_titles = [None] * nplots
            global_title = title
        xlabels = _normalize_seq(cfg.get("xlabel"), nplots)
        ylabels = _normalize_seq(cfg.get("ylabel"), nplots)
        rows, cols = self._grid_shape(cfg.get("grid"), nplots)
        page_size = rows * cols
        figures = []
        total_pages = (nplots + page_size - 1) // page_size

        for page_index in range(total_pages):
            start = page_index * page_size
            end = min(start + page_size, nplots)
            page_subplots = subplots[start:end]
            page_xlabels = xlabels[start:end]
            page_ylabels = ylabels[start:end]
            page_titles = per_titles[start:end]

            fig, axes = plt.subplots(
                rows,
                cols,
                figsize=cfg.get("figsize", (8.0, 6.0)),
                sharex=bool(cfg.get("sharex", False)),
                sharey=bool(cfg.get("sharey", False)),
                squeeze=False,
            )
            axes = axes.flatten()
            for idx, ax in enumerate(axes[: len(page_subplots)]):
                for series in page_subplots[idx]:
                    x = series.get("x")
                    y = series.get("y")
                    if x is None or y is None:
                        continue
                    ax.plot(x, y, label=series.get("label"))
                if page_ylabels[idx]:
                    ax.set_ylabel(page_ylabels[idx])
                if page_xlabels[idx]:
                    ax.set_xlabel(page_xlabels[idx])
                if page_titles[idx]:
                    ax.set_title(page_titles[idx])
                if cfg.get("grid", False):
                    ax.grid(True, alpha=0.3)
                if cfg.get("legend", True):
                    ax.legend(fontsize=9)
            for ax in axes[len(page_subplots) :]:
                ax.axis("off")
            if global_title:
                title_text = global_title
                if total_pages > 1:
                    title_text = f"{global_title} ({page_index + 1}/{total_pages})"
                fig.suptitle(title_text, fontsize=14, y=0.98)
            fig.tight_layout()
            page_cfg = dict(cfg)
            page_cfg["save"] = self._paged_save_target(cfg.get("save"), page_index, total_pages)
            figures.append(save_or_show(fig, page_cfg))

        return figures[0] if len(figures) == 1 else figures
