"""
Renderer for ``beeswarm_plot``.

**Usage context**

- Import these helpers from presentation workflows that produce tables, files, or plots.
- Reuse the public APIs here to keep output formatting and artifact behavior consistent.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from reaxkit.presentation.plot.renderers.base import PlotRenderer, merged, save_or_show


class BeeswarmPlotRenderer(PlotRenderer):
    """Render seaborn beeswarm (swarmplot) with value-based color mapping."""

    @staticmethod
    def _format_number(value) -> str:
        """Format a numeric annotation compactly and consistently."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "not recorded"
        return f"{number:.8g}" if np.isfinite(number) else "not recorded"

    def _render_diagnostic(self, cfg):
        """Render normalized parameter rows with objective-colored markers."""
        samples = pd.DataFrame(
            {
                "normalized_value": cfg.get("x", []),
                "plot_row": cfg.get("y", []),
                "objective_value": cfg.get("hue", []),
                "parameter_value": cfg.get("parameter_values", []),
                "parameter_label": cfg.get("parameter_labels", []),
                "sample_name": cfg.get("sample_names", []),
            }
        ).dropna(subset=["normalized_value", "plot_row", "objective_value"])
        parameter_payload = cfg.get("diagnostic_parameters")
        parameters = (
            parameter_payload.copy()
            if isinstance(parameter_payload, pd.DataFrame)
            else pd.DataFrame(parameter_payload or [])
        )
        if samples.empty or parameters.empty:
            fig, ax = plt.subplots(figsize=(7.6, 3.4))
            ax.text(0.5, 0.5, "No bounded diagnostic samples to plot", ha="center", va="center")
            ax.axis("off")
            return save_or_show(fig, cfg)

        parameters = parameters.sort_values("plot_row", kind="stable").reset_index(drop=True)
        row_count = len(parameters)
        fig = plt.figure(figsize=cfg.get("figsize", (14.0, max(4.8, 0.66 * row_count + 1.8))))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.11, right=0.54, top=0.90, bottom=0.13)

        cmap = LinearSegmentedColormap.from_list(
            "reaxkit_objective",
            ["#2563eb", "#f8fafc", "#dc2626"],
        )
        jitter_pattern = np.asarray([0.0, -0.16, 0.16, -0.08, 0.08, -0.22, 0.22], dtype=float)
        hover_artists: list[tuple[object, list[str]]] = []
        for parameter in parameters.to_dict(orient="records"):
            row_index = int(parameter["plot_row"])
            group = samples.loc[samples["plot_row"].astype(int) == row_index].reset_index(drop=True)
            if group.empty:
                continue
            color_min = float(parameter["color_min"])
            color_max = float(parameter["color_max"])
            norm = Normalize(vmin=color_min, vmax=color_max)
            y_values = row_index + jitter_pattern[np.arange(len(group)) % len(jitter_pattern)]
            artist = ax.scatter(
                pd.to_numeric(group["normalized_value"], errors="coerce"),
                y_values,
                c=pd.to_numeric(group["objective_value"], errors="coerce"),
                cmap=cmap,
                norm=norm,
                s=float(cfg.get("size", 42.0)),
                edgecolors=(0.06, 0.09, 0.16, 0.35),
                linewidths=0.6,
                zorder=3,
            )
            hover_artists.append((artist, [
                (
                    f"{label}\n{sample}: {self._format_number(value)}\n"
                    f"Objective function: {self._format_number(objective)}"
                )
                for label, sample, value, objective in zip(
                    group["parameter_label"],
                    group["sample_name"],
                    group["parameter_value"],
                    group["objective_value"],
                    strict=True,
                )
            ]))
            ax.hlines(row_index, 0.0, 1.0, color="#e2e8f0", linewidth=1.0, zorder=1)

        ax.set_xlim(-0.025, 1.025)
        ax.set_ylim(row_count - 0.45, -0.55)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0 (min)", "0.25", "0.50", "0.75", "1 (max)"])
        ax.set_yticks(
            parameters["plot_row"].astype(int).tolist(),
            parameters["parameter_key"].astype(str).tolist(),
        )
        ax.set_xlabel(cfg.get("xlabel", "Normalized parameter value within optimization bounds"))
        ax.set_ylabel(cfg.get("ylabel", "Parameter"))
        ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(str(cfg.get("title") or "Parameter diagnostics"), loc="left")

        fig.canvas.draw()
        axes_box = ax.get_position()
        annotation_x = axes_box.x1 + 0.025
        colorbar_x = 0.90
        fig.text(
            annotation_x,
            axes_box.y1 + 0.018,
            "Optimization bounds / final / starting",
            ha="left",
            va="bottom",
            fontsize=9,
            color="#475569",
        )
        for parameter in parameters.to_dict(orient="records"):
            row_index = int(parameter["plot_row"])
            _, row_y = fig.transFigure.inverted().transform(ax.transData.transform((0.0, row_index)))
            label = (
                f"[{self._format_number(parameter['lower_bound'])}, "
                f"{self._format_number(parameter['upper_bound'])}]   "
                f"Final: {self._format_number(parameter['final_value'])}   "
                f"Start: {self._format_number(parameter['starting_value'])}"
            )
            fig.text(annotation_x, row_y, label, ha="left", va="center", fontsize=8.5, color="#334155")

        global_scale = bool(cfg.get("global_objective_scale", False))
        if global_scale:
            color_min = float(parameters.iloc[0]["color_min"])
            color_max = float(parameters.iloc[0]["color_max"])
            colorbar_ax = fig.add_axes([
                colorbar_x,
                axes_box.y0 + 0.08 * axes_box.height,
                0.014,
                0.84 * axes_box.height,
            ])
            colorbar = fig.colorbar(
                ScalarMappable(norm=Normalize(vmin=color_min, vmax=color_max), cmap=cmap),
                cax=colorbar_ax,
            )
            colorbar.set_label("Objective function")
            colorbar.outline.set_visible(False)
        else:
            fig.text(colorbar_x, axes_box.y1 + 0.018, "Objective", ha="left", va="bottom", fontsize=9, color="#475569")
            row_height = axes_box.height / max(1, row_count)
            bar_height = min(0.065, row_height * 0.68)
            for parameter in parameters.to_dict(orient="records"):
                row_index = int(parameter["plot_row"])
                _, row_y = fig.transFigure.inverted().transform(ax.transData.transform((0.0, row_index)))
                color_min = float(parameter["color_min"])
                color_max = float(parameter["color_max"])
                colorbar_ax = fig.add_axes([colorbar_x, row_y - bar_height / 2.0, 0.010, bar_height])
                colorbar = ColorbarBase(
                    colorbar_ax,
                    cmap=cmap,
                    norm=Normalize(vmin=color_min, vmax=color_max),
                    orientation="vertical",
                    ticks=[color_min, color_max],
                )
                colorbar.ax.tick_params(labelsize=7, length=2)
                colorbar.outline.set_visible(False)

        tooltip = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#94a3b8", "alpha": 0.96},
            arrowprops={"arrowstyle": "->", "color": "#64748b"},
            fontsize=8.5,
        )
        tooltip.set_visible(False)

        def show_hover(event):
            visible = False
            if event.inaxes is ax:
                for artist, labels in hover_artists:
                    contains, info = artist.contains(event)
                    if not contains:
                        continue
                    point_index = int(info["ind"][0])
                    tooltip.xy = artist.get_offsets()[point_index]
                    tooltip.set_text(labels[point_index])
                    tooltip.set_visible(True)
                    visible = True
                    break
            if not visible and tooltip.get_visible():
                tooltip.set_visible(False)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", show_hover)
        return save_or_show(fig, cfg)

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
        from reaxkit.presentation.plot.renderers.beeswarm import BeeswarmPlotRenderer
        instance = BeeswarmPlotRenderer(...)
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
        if cfg.get("diagnostic_parameters") is not None:
            return self._render_diagnostic(cfg)
        import seaborn as sns

        x = cfg.get("x")
        y = cfg.get("y")
        hue = cfg.get("hue")
        if x is None or y is None:
            raise ValueError("beeswarm_plot requires 'x' and 'y'.")

        df = pd.DataFrame({"x": x, "y": y})
        if hue is None:
            hue = x
        df["hue"] = hue
        df = df.dropna(subset=["x", "y", "hue"])
        if df.empty:
            fig, ax = plt.subplots(figsize=(7.6, 3.4))
            ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
            ax.axis("off")
            return save_or_show(fig, cfg)

        fig = plt.figure(figsize=cfg.get("figsize", (9.2, max(3.4, 0.32 * df["y"].nunique()))))
        ax = fig.add_subplot(111)
        ax.axvline(0.0, c="grey", alpha=0.8, linewidth=1.0)
        marker_size = float(cfg.get("size", 3.5))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            ax = sns.swarmplot(
                data=df,
                x="x",
                y="y",
                hue="hue",
                palette=cfg.get("palette", "coolwarm"),
                size=marker_size,
                ax=ax,
            )
        placement_warning = any("cannot be placed" in str(w.message).lower() for w in caught)
        if placement_warning:
            ax.cla()
            ax.axvline(0.0, c="grey", alpha=0.8, linewidth=1.0)
            ax = sns.stripplot(
                data=df,
                x="x",
                y="y",
                hue="hue",
                palette=cfg.get("palette", "coolwarm"),
                size=max(2.5, marker_size - 0.8),
                jitter=float(cfg.get("jitter", 0.25)),
                alpha=float(cfg.get("alpha", 0.8)),
                ax=ax,
            )

        # Axis range: keep a readable near-1 window when sensitivity values cluster around 1.
        vals = pd.to_numeric(df["x"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size:
            xmin = float(np.nanmin(vals))
            xmax = float(np.nanmax(vals))
            p01 = float(np.nanpercentile(vals, 1))
            p99 = float(np.nanpercentile(vals, 99))
            median = float(np.nanmedian(vals))
            span = max(p99 - p01, xmax - xmin)
            if 0.95 <= median <= 1.05 and span <= 0.35:
                ax.set_xlim(0.9, 1.1)
            else:
                pad = max(0.02 * max(abs(p01), abs(p99), 1.0), 0.01)
                ax.set_xlim(p01 - pad, p99 + pad)

        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        lower_handle = Line2D([0], [0], marker="o", color="w", markerfacecolor="#3B4CC0", markersize=7, label="Lower parameter value")
        higher_handle = Line2D([0], [0], marker="o", color="w", markerfacecolor="#B40426", markersize=7, label="Higher parameter value")
        ax.legend(
            handles=[lower_handle, higher_handle],
            loc=str(cfg.get("legend_loc", "best")),
            frameon=False,
            title=str(cfg.get("legend_title", "Color Meaning")),
        )
        ax.spines["left"].set_visible(True)
        ax.grid(axis="x")
        ax.set_xlabel(cfg.get("xlabel", "Value"))
        ax.set_ylabel(cfg.get("ylabel", ""))
        title = cfg.get("title")
        if title:
            ax.set_title(str(title))
        fig.tight_layout()
        return save_or_show(fig, cfg)
