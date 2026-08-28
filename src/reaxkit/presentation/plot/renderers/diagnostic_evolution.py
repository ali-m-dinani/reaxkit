"""Renderer for per-parameter force-field diagnostic evolution plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap, Normalize

from reaxkit.presentation.plot.renderers.base import PlotRenderer, merged, save_or_show


class DiagnosticEvolutionPlotRenderer(PlotRenderer):
    """Render one objective-colored scatter-and-line axis per parameter."""

    @staticmethod
    def _format_number(value) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "not recorded"
        return f"{number:.8g}" if np.isfinite(number) else "not recorded"

    def render(self, result, style=None):
        cfg = merged(result, style)
        samples = pd.DataFrame(
            {
                "epoch": cfg.get("x", []),
                "normalized_value": cfg.get("y", []),
                "objective_value": cfg.get("hue", []),
                "parameter_value": cfg.get("parameter_values", []),
                "parameter_key": cfg.get("parameter_keys", []),
                "parameter_label": cfg.get("parameter_labels", []),
            }
        ).dropna(subset=["epoch", "normalized_value", "objective_value", "parameter_key"])
        parameter_payload = cfg.get("diagnostic_parameters")
        parameters = (
            parameter_payload.copy()
            if isinstance(parameter_payload, pd.DataFrame)
            else pd.DataFrame(parameter_payload or [])
        )
        if samples.empty or parameters.empty:
            fig, ax = plt.subplots(figsize=(8.0, 3.5))
            ax.text(0.5, 0.5, "No parameter evolution samples to plot", ha="center", va="center")
            ax.axis("off")
            return save_or_show(fig, cfg)

        parameters = parameters.sort_values("plot_row", kind="stable").reset_index(drop=True)
        row_count = len(parameters)
        fig, axes = plt.subplots(
            row_count,
            1,
            squeeze=False,
            figsize=cfg.get("figsize", (14.0, max(4.8, 2.55 * row_count))),
        )
        axes = axes[:, 0]
        fig.subplots_adjust(left=0.10, right=0.60, top=0.93, bottom=0.08, hspace=0.62)
        cmap = LinearSegmentedColormap.from_list(
            "reaxkit_objective",
            ["#2563eb", "#f8fafc", "#dc2626"],
        )
        global_scale = bool(cfg.get("global_objective_scale", False))

        for axis, parameter in zip(axes, parameters.to_dict(orient="records"), strict=True):
            key = str(parameter["parameter_key"])
            group = samples.loc[samples["parameter_key"].astype(str) == key].copy()
            group["epoch"] = pd.to_numeric(group["epoch"], errors="coerce")
            group = group.dropna(subset=["epoch"]).sort_values("epoch", kind="stable")
            objectives = pd.to_numeric(group["objective_value"], errors="coerce").to_numpy(dtype=float)
            color_min = float(parameter["color_min"])
            color_max = float(parameter["color_max"])
            if not global_scale:
                finite_objectives = objectives[np.isfinite(objectives)]
                if finite_objectives.size:
                    color_min = float(np.min(finite_objectives))
                    color_max = float(np.max(finite_objectives))
                    if color_min == color_max:
                        padding = max(abs(color_min) * 1e-6, 1e-9)
                        color_min -= padding
                        color_max += padding
            norm = Normalize(vmin=color_min, vmax=color_max)
            epochs = group["epoch"].to_numpy(dtype=float)
            normalized = pd.to_numeric(group["normalized_value"], errors="coerce").to_numpy(dtype=float)
            axis.plot(epochs, normalized, color="#64748b", linewidth=1.15, zorder=2)
            axis.scatter(
                epochs,
                normalized,
                c=objectives,
                cmap=cmap,
                norm=norm,
                s=float(cfg.get("size", 46.0)),
                edgecolors=(0.06, 0.09, 0.16, 0.40),
                linewidths=0.6,
                zorder=3,
            )
            axis.set_title(f"{key}  {parameter.get('parameter_label', '')}", loc="left", fontsize=10)
            axis.set_ylabel("Normalized value")
            axis.grid(color="#e2e8f0", linewidth=0.8)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.text(
                1.025,
                0.62,
                (
                    f"Bounds: [{self._format_number(parameter.get('lower_bound'))}, "
                    f"{self._format_number(parameter.get('upper_bound'))}]\n"
                    f"Start: {self._format_number(parameter.get('starting_value'))}\n"
                    f"End: {self._format_number(parameter.get('final_value'))}"
                ),
                transform=axis.transAxes,
                ha="left",
                va="center",
                fontsize=8.5,
                color="#334155",
            )
            axis_box = axis.get_position()
            colorbar_ax = fig.add_axes([0.89, axis_box.y0 + 0.08 * axis_box.height, 0.010, 0.84 * axis_box.height])
            colorbar = ColorbarBase(
                colorbar_ax,
                cmap=cmap,
                norm=norm,
                orientation="vertical",
                ticks=[color_min, color_max],
            )
            colorbar.ax.tick_params(labelsize=7, length=2)
            colorbar.outline.set_visible(False)

        axes[-1].set_xlabel(cfg.get("xlabel", "Epoch"))
        fig.suptitle(str(cfg.get("title") or "Parameter evolution"), x=0.10, ha="left", fontsize=13)
        fig.text(0.89, 0.95, "Objective", ha="left", va="bottom", fontsize=9, color="#475569")
        return save_or_show(fig, cfg)


__all__ = ["DiagnosticEvolutionPlotRenderer"]
