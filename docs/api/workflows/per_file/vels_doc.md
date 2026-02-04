# Vels Workflow

CLI namespace: `reaxkit vels <task> [flags]`

Velocity and acceleration (vels) analysis workflow for ReaxKit.

This workflow provides tools for inspecting, visualizing, and exporting data
from ReaxFF velocity-related output files, including `vels`, `moldyn.vel`,
and `molsav`.

It supports:
- Extracting per-atom coordinates, velocities, accelerations, or previous-step
  accelerations for all atoms or selected subsets.
- Printing extracted data to the console or exporting it to CSV.
- Visualizing scalar velocity or acceleration components (e.g. vx, vz, ax)
  mapped onto atomic positions using 3D scatter plots.
- Generating 2D projected heatmaps (xy, xz, yz) of scalar quantities by spatial
  binning and aggregation.

The workflow is designed for spatially resolved analysis of atomic motion and
dynamics in ReaxFF molecular dynamics simulations.

## Available tasks

### `get`

#### Examples

- `reaxkit vels get --key velocities --atoms 1,3,7 --print`
- `reaxkit vels get --key coordinates --atoms 1-50 --export coords.csv`
- `reaxkit vels get --key metadata --print`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to vels/moldyn.vel/molsav file. |
| `--atoms ATOMS` | 1-based atom indices (optional). Examples: "1,3,7" or "1-10,25". |
| `--export EXPORT` | Path to export CSV data. |
| `--print` | Print output to console. |
| `--key {metadata,coordinates,velocities,accelerations,prev_accelerations}` | Which dataset to return. |

### `heatmap2d`

#### Examples

- `reaxkit vels heatmap2d --value vz --plane xz --bins 60 --agg mean --save vz_xz.png`
- `reaxkit vels heatmap2d --value vz --plane xy --bins 80,40 --agg max`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to vels/moldyn.vel/molsav file. |
| `--atoms ATOMS` | 1-based atom indices. Examples: "1,3,7" or "1-10,25". |
| `--value VALUE` | Scalar to plot: vx,vy,vz, ax,ay,az, pax,pay,paz. |
| `--plane {xy,xz,yz}` | Projection plane. |
| `--bins BINS` | Grid bins: "N" or "Nx,Ny" (e.g., "80,40"). |
| `--agg AGG` | Aggregation: mean\|max\|min\|sum\|count. |
| `--save SAVE` | Path to save plot image (dir or full filename). |
| `--vmin VMIN` | Color scale min (auto if not set). |
| `--vmax VMAX` | Color scale max (auto if not set). |
| `--cmap CMAP` | Matplotlib colormap. |

### `plot3d`

#### Examples

- `reaxkit vels plot3d --value vz --save vz_3d.png`
- `reaxkit vels plot3d --value vz --atoms 1-500 --cmap coolwarm`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to vels/moldyn.vel/molsav file. |
| `--atoms ATOMS` | 1-based atom indices. Examples: "1,3,7" or "1-10,25". |
| `--value VALUE` | Scalar to plot: vx,vy,vz, ax,ay,az, pax,pay,paz. |
| `--save SAVE` | Path to save plot image (dir or full filename). |
| `--vmin VMIN` | Color scale min (auto if not set). |
| `--vmax VMAX` | Color scale max (auto if not set). |
| `--cmap CMAP` | Matplotlib colormap. |
| `--size SIZE` | Marker size. |
| `--alpha ALPHA` | Marker transparency. |
| `--elev ELEV` | 3D view elevation. |
| `--azim AZIM` | 3D view azimuth. |
