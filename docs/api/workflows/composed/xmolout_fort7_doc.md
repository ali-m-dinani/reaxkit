# Xmolout Fort7 Workflow

CLI namespace: `reaxkit xmolout_fort7 <task> [flags]`

xmolout–fort.7 combined workflow for ReaxKit.

This workflow provides visualization and analysis tasks that require both
atomic coordinates from `xmolout` and per-atom properties from `fort.7`
(e.g. partial charges, bond-order–derived quantities).

It supports:
- 3D scatter visualization of arbitrary fort.7 scalar properties mapped onto
  atomic positions across selected frames.
- 2D projected heatmaps (xy/xz/yz) of atomic properties or atom counts,
  with flexible spatial binning and aggregation.

The workflow is designed for spatially resolved analysis of ReaxFF outputs,
enabling intuitive inspection of per-atom quantities in real space and
across time.

## Available tasks

### `heatmap2d`

#### Examples

- `reaxkit xmolfort7 heatmap2d --property partial_charge --plane xz --bins 10 --agg mean --frames 0:300:100`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--xmolout XMOLOUT` | Path to xmolout file. |
| `--fort7 FORT7` | Path to fort.7 file. |
| `--plot` | Show plot interactively. |
| `--export EXPORT` | Path to export CSV data. |
| `--frames FRAMES` | Frames: "0,10,20" or "0:100:5". |
| `--plane {xy,xz,yz}` | Projection plane. |
| `--bins BINS` | Grid bins: "N" or "Nx,Ny" (e.g., "10,25"). |
| `--agg AGG` | Aggregation: mean\|max\|min\|sum\|count. |
| `--property PROPERTY` | fort7 column or alias to aggregate (e.g., partial_charge\|charge\|q). |
| `--vmin VMIN` | Color scale min (auto if not set). |
| `--vmax VMAX` | Color scale max (auto if not set). |
| `--cmap CMAP` | Matplotlib colormap. |
| `--save SAVE` | Path to save plot image. |

### `plot3d`

#### Examples

- `reaxkit xmolfort7 plot3d --property charge --frames 0:20:10`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--xmolout XMOLOUT` | Path to xmolout file. |
| `--fort7 FORT7` | Path to fort.7 file. |
| `--plot` | Show plot interactively. |
| `--export EXPORT` | Path to export CSV data. |
| `--property PROPERTY` | Column name or alias (e.g., partial_charge, charge, q). |
| `--frames FRAMES` | Frames: "0,10,20" or "0:100:5". |
| `--atoms ATOMS` | Atom indices: "0,1,2" (0-based). |
| `--vmin VMIN` | Color scale min (auto if not set). |
| `--vmax VMAX` | Color scale max (auto if not set). |
| `--size SIZE` | Marker size. |
| `--alpha ALPHA` | Marker transparency. |
| `--cmap CMAP` | Matplotlib colormap. |
| `--elev ELEV` | 3D view elevation. |
| `--azim AZIM` | 3D view azimuth. |
| `--save SAVE` | Path to save plot image. |
