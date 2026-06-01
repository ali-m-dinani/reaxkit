"""
Example: presentation plotting utilities aligned with `gen-plot`.

This example shows how to use ReaxKit plotting utilities directly on
tabular data, without a ReaxFF-specific workflow.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reaxkit.presentation.plot import (
    directed_plot,
    dual_yaxis_plot,
    heatmap2d_from_3d,
    scatter3d_points,
    single_plot,
)


HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "data" / "sample_tabular_data.csv"

OUTDIR = Path("reaxkit_outputs/examples/gen_plot_presentation")
OUTDIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Missing example data file: {DATA_FILE}")

try:
    df = pd.read_csv(DATA_FILE, comment="#", header=None)
except Exception:
    df = pd.read_csv(DATA_FILE, comment="#", delim_whitespace=True, header=None)

df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all").reset_index(drop=True)
print("Loaded data shape:", df.shape)
print(df.head())

if df.shape[1] < 3:
    raise ValueError("sample_tabular_data.csv must provide at least 3 numeric columns.")

x = df.iloc[:, 0].to_numpy()
y1 = df.iloc[:, 1].to_numpy()
y2 = df.iloc[:, 2].to_numpy()

single_plot(
    series=[{"x": x, "y": y1, "label": "c2 vs c1"}],
    title="Single plot c2 vs c1",
    xlabel="c1",
    ylabel="c2",
    save=OUTDIR / "single_plot.png",
)

single_plot(
    series=[
        {"x": x, "y": y1, "label": "c2"},
        {"x": x, "y": y2, "label": "c3"},
    ],
    title="Multiple series c2 & c3 vs c1",
    xlabel="c1",
    ylabel="value",
    legend=True,
    save=OUTDIR / "multi_series_plot.png",
)

directed_plot(
    x=x,
    y=y1,
    title="Directed plot progression along c2 vs c1",
    xlabel="c1",
    ylabel="c2",
    save=OUTDIR / "directed_plot.png",
)

dual_yaxis_plot(
    x=x,
    y1=y1,
    y2=y2,
    title="Dual y-axis c2 and c3 vs c1",
    xlabel="c1",
    ylabel1="c2",
    ylabel2="c3",
    save=OUTDIR / "dual_yaxis_plot.png",
)

if df.shape[1] >= 4:
    coords = df.iloc[:, 0:3].to_numpy()
    values = df.iloc[:, 3].to_numpy()

    scatter3d_points(
        coords=coords,
        values=values,
        title="3D scatter c1,c2,c3 colored by c4",
        save=OUTDIR / "scatter3d.png",
    )

    heatmap2d_from_3d(
        coords=coords,
        values=values,
        plane="xz",
        bins=(60, 40),
        title="2D heatmap (xz plane)",
        save=OUTDIR / "heatmap_xz.png",
    )

print(f"\nDone. Plots written to: {OUTDIR.resolve()}")
