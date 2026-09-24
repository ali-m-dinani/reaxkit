"""Plot exports for tabular ferroelectric switching-model fits."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def _safe_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("._") or "group"


def _plot_imports():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, Normalize
    except ImportError as exc:
        raise RuntimeError(
            "Switching-fit plots require matplotlib. Install ReaxKit with its plot "
            "extra: pip install 'reaxkit[plot]'."
        ) from exc
    return plt, LinearSegmentedColormap, Normalize


def _blue_red_colormap(LinearSegmentedColormap):
    return LinearSegmentedColormap.from_list(
        "switching_blue_red",
        ("#08306b", "#225ea8", "#6a51a3", "#b2182b", "#7f0000"),
    )


def _base_axis(ax, *, model: str, group: object, time_unit: str) -> None:
    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_ylabel("Flipped fraction")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"{model.upper()} switching fit - group {group}")
    ax.grid(alpha=0.2)


def _draw_observations(ax, table) -> None:
    ax.plot(
        table["time"],
        table["observed_fraction"],
        linestyle="None",
        marker="o",
        markersize=4.5,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.9,
        label="data",
        zorder=5,
    )


def _draw_best_fit(ax, table) -> None:
    ax.plot(
        table["time"],
        table["predicted_fraction"],
        color="#15803d",
        linewidth=2.8,
        label="best fit",
        zorder=4,
    )


def generate_switching_plots(
    result,
    output_directory: Path,
    *,
    time_unit: str = "input units",
    dpi: int = 180,
) -> list[Path]:
    """Write fit, residual, sweep, and model-comparison PNG plots.

    Observations use markers without connecting lines. Parameter-sweep curves
    progress from dark blue at the smallest value to dark red at the largest;
    the unconstrained best fit is always green.
    """
    if dpi <= 0:
        raise ValueError("figure_dpi must be greater than zero.")
    plt, LinearSegmentedColormap, Normalize = _plot_imports()
    cmap = _blue_red_colormap(LinearSegmentedColormap)
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for (group, model), best in result.fitted_curves.groupby(
        ["group", "model"], sort=False, dropna=False
    ):
        group_name = _safe_name(group)
        model_name = _safe_name(model)
        relevant_sweeps = result.sweep_curves[
            (result.sweep_curves["model"] == model)
        ]
        if not relevant_sweeps.empty:
            if isinstance(group, float) and np.isnan(group):
                relevant_sweeps = relevant_sweeps[relevant_sweeps["group"].isna()]
            else:
                relevant_sweeps = relevant_sweeps[relevant_sweeps["group"] == group]

        if relevant_sweeps.empty:
            fig, ax = plt.subplots(figsize=(8.2, 5.4))
            _draw_observations(ax, best)
            _draw_best_fit(ax, best)
            _base_axis(ax, model=str(model), group=group, time_unit=time_unit)
            ax.legend(frameon=False)
            fig.tight_layout()
            path = destination / f"{group_name}_{model_name}_best_fit.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            written.append(path)
        else:
            for parameter, sweep in relevant_sweeps.groupby(
                "sweep_parameter", sort=False, dropna=False
            ):
                values = np.sort(sweep["sweep_value"].unique().astype(float))
                normalizer = Normalize(vmin=float(values[0]), vmax=float(values[-1]))
                if values.size == 1:
                    normalizer = Normalize(
                        vmin=float(values[0]) - 0.5,
                        vmax=float(values[0]) + 0.5,
                    )
                fig, ax = plt.subplots(figsize=(8.8, 5.8))
                _draw_observations(ax, best)
                for value in values:
                    curve = sweep[np.isclose(sweep["sweep_value"], value)]
                    ax.plot(
                        curve["time"],
                        curve["predicted_fraction"],
                        color=cmap(normalizer(value)),
                        linewidth=1.6,
                        alpha=0.95,
                        label=f"{parameter}={value:g}",
                    )
                _draw_best_fit(ax, best)
                _base_axis(ax, model=str(model), group=group, time_unit=time_unit)
                if values.size <= 12:
                    ax.legend(frameon=False, ncol=2, fontsize=8)
                else:
                    ax.legend(handles=ax.lines[:1] + ax.lines[-1:], frameon=False)
                    scalar = plt.cm.ScalarMappable(norm=normalizer, cmap=cmap)
                    fig.colorbar(scalar, ax=ax, label=str(parameter))
                fig.tight_layout()
                path = destination / (
                    f"{group_name}_{model_name}_{_safe_name(parameter)}_sweep.png"
                )
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                written.append(path)

        fig, ax = plt.subplots(figsize=(8.2, 3.8))
        ax.axhline(0.0, color="#555555", linewidth=1.0)
        ax.plot(
            best["time"],
            best["residual"],
            linestyle="None",
            marker="o",
            markersize=4.0,
            color="#15803d",
        )
        ax.set_xlabel(f"Time ({time_unit})")
        ax.set_ylabel("Fit residual")
        ax.set_title(f"{str(model).upper()} best-fit residuals - group {group}")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = destination / f"{group_name}_{model_name}_residuals.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    for group, curves in result.fitted_curves.groupby("group", sort=False, dropna=False):
        if curves["model"].nunique() < 2:
            continue
        fig, ax = plt.subplots(figsize=(8.8, 5.8))
        first_model = next(iter(curves.groupby("model", sort=False)))[1]
        _draw_observations(ax, first_model)
        for model, curve in curves.groupby("model", sort=False):
            ax.plot(
                curve["time"],
                curve["predicted_fraction"],
                linewidth=2.1,
                label=f"{str(model).upper()} best fit",
            )
        ax.set_xlabel(f"Time ({time_unit})")
        ax.set_ylabel("Flipped fraction")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"Best switching-model fits - group {group}")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = destination / f"{_safe_name(group)}_model_comparison.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


__all__ = ["generate_switching_plots"]
