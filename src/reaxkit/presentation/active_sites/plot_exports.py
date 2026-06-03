"""Export TRACT-style diagnostic figures for active-site structural outputs.

This module renders PNG maps/histograms from active-site structural tables for
reporting and visual QA. It is scoped to plotting/export and does not compute
structural descriptors.

**Usage context**

- Figure generation: Create TRACT-style structural maps and distributions.
- Report assembly: Produce image assets consumed by report payload builders.
- QA workflows: Visualize labels, grains, and pyramidalization patterns.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _has_columns(df: pd.DataFrame, cols: list[str]) -> bool:
    """
    Has columns.
    """
    return all(c in df.columns for c in cols)


def _plot_dpyr_map(df: pd.DataFrame, outpath: Path) -> None:
    """
    Plot dpyr map.
    """
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
    """
    Plot label map.
    """
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
    """
    Plot grain map.
    """
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
    """
    Plot dpyr hist.
    """
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
    """Write TRACT-style structural PNG figures and return written filenames.

    Parameters
    -----
    table : pd.DataFrame
        Active-site structural table containing required plotting columns.
    out_dir : Path
        Output directory where figures will be written.
    stem : str
        File stem prefix used to build output PNG names.

    Returns
    -----
    list[str]
        Filenames successfully written into `out_dir`.

    Examples
    -----
    ```python
    files = save_structural_figures_tract_style(table, Path("analysis"), stem="frame0000")
    ```
    Sample output:
    `["frame0000_dpyr_map.png", "frame0000_label_map.png", ...]`
    Meaning:
    Generated figures are available for embedding in structural reports.
    """
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


def save_event_diagnostic_figures(
    distance_table: pd.DataFrame,
    episode_table: pd.DataFrame,
    summary: dict,
    out_dir: Path,
) -> list[str]:
    """Write TRACT-style diagnostic event figures and return filenames."""
    if not isinstance(distance_table, pd.DataFrame) or distance_table.empty:
        return []
    if not isinstance(summary, dict) or not bool(summary.get("diagnostic", False)):
        return []
    required = ["species", "min_distance"]
    if not _has_columns(distance_table, required):
        return []

    species_available = [
        species
        for species in ("C-O", "C-Si")
        if not distance_table[distance_table["species"] == species].empty
    ]
    if not species_available:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    n_panels = 2 * len(species_available)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    ax_idx = 0
    species_summary = summary.get("species", {}) if isinstance(summary, dict) else {}
    eq_bonds = {"C-O": 1.43, "C-Si": 1.88}
    r_probe = float(summary.get("r_probe", 2.5)) if isinstance(summary, dict) else 2.5
    frames_analyzed = int(summary.get("frames_analyzed", 0)) if isinstance(summary, dict) else 0
    every = int(summary.get("every", 1)) if isinstance(summary, dict) else 1

    for species in species_available:
        dists = pd.to_numeric(
            distance_table.loc[distance_table["species"] == species, "min_distance"],
            errors="coerce",
        ).dropna()
        dists = dists[dists < 3.5]

        ax = axes[ax_idx]
        ax_idx += 1
        ax.hist(dists, bins=80, color="steelblue", alpha=0.7, edgecolor="none")
        ax.set_yscale("log")
        ax.axvline(eq_bonds.get(species, 0.0), color="green", linestyle="--", linewidth=1.2, label=f"Eq. bond {eq_bonds.get(species, 0.0):.2f} A")
        ax.axvline(r_probe, color="orange", linestyle=":", linewidth=1.2, label=f"r_probe {r_probe:.2f} A")
        suggested = species_summary.get(species, {}).get("suggested_r_cut") if isinstance(species_summary, dict) else None
        if suggested is not None:
            ax.axvline(float(suggested), color="red", linestyle="-", linewidth=1.5, label=f"Valley -> r_cut ~= {float(suggested):.2f} A")
        ax.set_xlabel(f"Min {species} distance [A]")
        ax.set_ylabel("Count (log scale)")
        ax.set_title(f"{species} distance distribution\n({frames_analyzed} frames, {len(dists)} C-atom samples)")
        ax.legend(fontsize=8)

        ax = axes[ax_idx]
        ax_idx += 1
        if isinstance(episode_table, pd.DataFrame) and not episode_table.empty and {"species", "duration_ps"}.issubset(episode_table.columns):
            durations = pd.to_numeric(
                episode_table.loc[episode_table["species"] == species, "duration_ps"],
                errors="coerce",
            ).dropna()
        else:
            durations = pd.Series(dtype=float)
        if not durations.empty:
            ax.hist(durations, bins=40, color="coral", alpha=0.8, edgecolor="none")
            ax.axvline(5.0, color="red", linestyle="--", linewidth=1.5, label="Current persist = 5 ps")
            ax.set_xlabel(f"{species} episode duration [ps]")
            ax.set_ylabel("Count")
            ax.set_title(f"{species} close-approach episode lengths\n(r_probe < {r_probe:.2f} A, {len(durations)} episodes)")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, f"No {species} episodes\nbelow {r_probe:.2f} A", ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{species} episode durations")

    for j in range(ax_idx, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"TRACT Tool 2 - Diagnostic ({frames_analyzed} frames analyzed, stride={every})", fontsize=11, y=1.01)
    fig.tight_layout()
    path = out_dir / "diagnose_distances.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path.name] if path.exists() else []


__all__ = ["save_structural_figures_tract_style", "save_event_diagnostic_figures"]
