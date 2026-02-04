"""
Example: Meta plotting with plotter.py

This example shows how to use ReaxKit's plotter utilities directly
(on any tabular text file), without going through a ReaxFF-specific
workflow.

Input file:
    sample_tabular_data.csv

Expected format:
    - whitespace-, CSV-, or TSV-delimited numeric columns
    - optional comment lines starting with '#'
    - no strict header requirements

Column convention (1-based, for explanation only):
    c1 -> column 1
    c2 -> column 2
    c3 -> column 3
    c4 -> column 4
"""

from pathlib import Path
import pandas as pd

from reaxkit.utils.media.plotter import (
    single_plot,
    directed_plot,
    dual_yaxis_plot,
    scatter3d_points,
    heatmap2d_from_3d,
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "data" / "sample_tabular_data.csv"

OUTDIR = Path("reaxkit_outputs/examples/plotter")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Load tabular data
# ------------------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Missing example data file: {DATA_FILE}")

# Flexible read: whitespace or CSV both work
try:
    df = pd.read_csv(DATA_FILE, comment="#", header=None)
except Exception:
    df = pd.read_csv(DATA_FILE, comment="#", delim_whitespace=True, header=None)

df = df.apply(pd.to_numeric, errors="coerce")

print("Loaded data shape:", df.shape)
print(df.head())

# Column mapping (0-based internally)
x = df.iloc[:, 0].to_numpy()
y1 = df.iloc[:, 1].to_numpy()
y2 = df.iloc[:, 2].to_numpy()


# ------------------------------------------------------------
# 1) Simple single-curve plot
# ------------------------------------------------------------

single_plot(
    x=x,
    y=y1,
    title="Single plot c2 vs c1",
    xlabel="c1",
    ylabel="c2",
    save=OUTDIR,
)


# ------------------------------------------------------------
# 2) Multi-series single plot
# ------------------------------------------------------------

single_plot(
    series=[
        {"x": x, "y": y1, "label": "c2"},
        {"x": x, "y": y2, "label": "c3"},
    ],
    title="Multiple series c2 & c3 vs c1",
    xlabel="c1",
    ylabel="value",
    legend=True,
    save=OUTDIR,
)


# ------------------------------------------------------------
# 3) Directed plot (trajectory-style)
# ------------------------------------------------------------

directed_plot(
    x=x,
    y=y1,
    title="Directed plot progression along c2 vs c1",
    xlabel="c1",
    ylabel="c2",
    save=OUTDIR,
)


# ------------------------------------------------------------
# 4) Dual y-axis plot
# ------------------------------------------------------------

dual_yaxis_plot(
    x=x,
    y1=y1,
    y2=y2,
    title="Dual y-axis c2 and c3 vs c1",
    xlabel="c1",
    ylabel1="c2",
    ylabel2="c3",
    save=OUTDIR,
)


# ------------------------------------------------------------
# 5) 3D scatter plot (if file has ≥ 4 columns)
# ------------------------------------------------------------

if df.shape[1] >= 4:
    # skipping the headers by selecting [1:, ...]
    coords = df.iloc[1:, 2:5].to_numpy()    # c1, c2, c3 as (x,y,z)
    values = df.iloc[1:, 5].to_numpy()     # c4 as scalar

    scatter3d_points(
        coords=coords,
        values=values,
        title="3D scatter c1,c2,c3 colored by c4",
        save=OUTDIR,
    )

    # --------------------------------------------------------
    # 6) 2D heatmap projection from 3D data
    # --------------------------------------------------------

    heatmap2d_from_3d(
        coords=coords,
        values=values,
        plane="xz",
        bins=(60, 40),
        title="2D heatmap (xz plane)",
        save=OUTDIR,
    )


print(f"\n✅ Plots written to: {OUTDIR.resolve()}")
