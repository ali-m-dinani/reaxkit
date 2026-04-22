"""TRACT-style figure exports for active-site structural outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _has_columns(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)


def _plot_dpyr_map(df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    regular = df[~df["is_undercoord"]]
    under = df[df["is_undercoord"]]

    sc = ax.scatter(
        regular["x"],
        regular["y"],
        c=regular["d_pyr"].abs(),
        cmap="plasma",
        s=6,
        vmin=0.0,
        vmax=0.5,
        alpha=0.85,
    )
    ax.scatter(
        under["x"],
        under["y"],
        marker="*",
        c="red",
        s=50,
        zorder=5,
        label="under-coord (Ea~=0)",
    )
    cb = fig.colorbar(sc, ax=ax, label="|d_pyr| [A]")
    cb.ax.axhline(0.229, color="cyan", linewidth=1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [A]")
    ax.set_ylabel("y [A]")
    ax.set_title("Pyramidalization map (|d_pyr|)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_label_map(df: pd.DataFrame, outpath: Path) -> None:
    color_map = {
        "basal": "#4a4a8a",
        "edge_zigzag": "#e06c00",
        "edge_armchair": "#c8a200",
        "under_coordinated": "#d62728",
        "defect": "#2ca02c",
        "interior": "#aaaaaa",
        "other": "#888888",
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl, grp in df.groupby("label"):
        ax.scatter(grp["x"], grp["y"], c=color_map.get(str(lbl), "#888888"), s=5, label=str(lbl), alpha=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x [A]")
    ax.set_ylabel("y [A]")
    ax.set_title("Atom labels")
    ax.legend(fontsize=7, markerscale=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_grain_map(df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    grain_ids = df["grain_id"].to_numpy(dtype=int)
    unique_grains = sorted(set(grain_ids.tolist()))
    cmap = plt.get_cmap("tab20", max(len(unique_grains), 2))
    for gi in unique_grains:
        grp = df[df["grain_id"] == gi]
        color = "#cccccc" if gi == -1 else cmap(unique_grains.index(gi))
        ax.scatter(grp["x"], grp["y"], c=[color], s=5, alpha=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x [A]")
    ax.set_ylabel("y [A]")
    n_grains = len([g for g in unique_grains if g >= 0])
    ax.set_title(f"Grain map ({n_grains} grains)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_dpyr_hist(df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for lbl, color in [
        ("basal", "#4a4a8a"),
        ("defect", "#2ca02c"),
        ("edge_zigzag", "#e06c00"),
        ("edge_armchair", "#c8a200"),
    ]:
        grp = df[(df["label"] == lbl) & (~df["is_undercoord"])]
        if len(grp) > 0:
            ax.hist(
                grp["d_pyr"].abs(),
                bins=40,
                range=(0.0, 0.6),
                alpha=0.55,
                label=lbl,
                color=color,
                density=True,
            )
    ax.axvline(0.229, color="black", linestyle="--", linewidth=1.5, label="tau_opt = 0.229 A")
    ax.set_xlabel("|d_pyr| [A]")
    ax.set_ylabel("Density")
    ax.set_title("Pyramidalization distribution by label")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def save_structural_figures_tract_style(
    table: pd.DataFrame,
    out_dir: Path,
    *,
    stem: str,
) -> list[str]:
    """Write TRACT-style structural PNG figures and return written filenames."""
    if not isinstance(table, pd.DataFrame) or table.empty:
        return []
    required = ["x", "y", "d_pyr", "is_undercoord", "label", "grain_id"]
    if not _has_columns(table, required):
        return []

    df = table.loc[:, required].copy()
    if df.isna().all(axis=None):
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    files = {
        "dpyr_map": out_dir / f"{stem}_dpyr_map.png",
        "label_map": out_dir / f"{stem}_label_map.png",
        "grain_map": out_dir / f"{stem}_grain_map.png",
        "dpyr_hist": out_dir / f"{stem}_dpyr_hist.png",
    }
    _plot_dpyr_map(df, files["dpyr_map"])
    _plot_label_map(df, files["label_map"])
    _plot_grain_map(df, files["grain_map"])
    _plot_dpyr_hist(df, files["dpyr_hist"])

    for path in files.values():
        if path.exists():
            written.append(path.name)
    return written


__all__ = ["save_structural_figures_tract_style"]
