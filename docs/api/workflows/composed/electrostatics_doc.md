# Electrostatics Workflow

CLI namespace: `reaxkit electrostatics <task> [flags]`

Electrostatics workflow for ReaxKit.

This workflow provides tools for computing and analyzing electrostatic properties
from ReaxFF simulations, including dipole moments, polarizations, and
polarization–electric-field hysteresis behavior.

It supports:
- Dipole moment and polarization calculations for single frames, at both
  total-system and local (core-atom cluster) levels.
- Polarization–field hysteresis analysis using time-dependent electric fields
  (via fort.78 and control files), including extraction of coercive fields and
  remnant polarizations.
- Visualization of local electrostatics through 3D scatter plots and 2D
  projected heatmaps based on atomic coordinates.

The workflow integrates xmolout, fort.7, fort.78, and control files, and is
designed to bridge atomistic ReaxFF simulations with experiment-facing
electrostatic observables.

## Available tasks

### `dipole`

#### Examples

- `reaxkit elect dipole --frame 10 --scope total --export dipole_pol_total_frame10.csv --polarization`
- `reaxkit elect dipole --frame 10 --scope local --core Al --export dipole_pol_local_frame10.csv --polarization`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--xmolout XMOLOUT` | Path to xmolout file |
| `--fort7 FORT7` | Path to fort.7 file |
| `--frame FRAME` | 0-based frame index in xmolout |
| `--scope {total,local}` | Electrostatics scope: total or local |
| `--core CORE` | Comma-separated core atom types for local scope (e.g. Al,Mg) |
| `--export EXPORT` | CSV file to export the dipole/polarization data |
| `--polarization` | If present, compute polarization instead of dipole |

### `heatmap2d`

#### Examples

- `reaxkit elect heatmap2d --core Al --component mu_z --plane xz --bins 10 --frames 0:2:1 --save reaxkit_outputs/elect/local_mu_2d`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--xmolout XMOLOUT` | Path to xmolout file |
| `--fort7 FORT7` | Path to fort.7 file |
| `--core CORE` | Comma-separated core atom types (e.g. Al,Mg) |
| `--component COMPONENT` | Which component to aggregate (e.g. pol_z, P_z (uC/cm^2), mu_z) |
| `--frames FRAMES` | Frames: "0,10,20" or "0:100:5" |
| `--polarization` | Use polarization components (P_x/y/z) instead of dipole |
| `--plane {xy,xz,yz}` | Projection plane |
| `--bins BINS` | Grid bins: "N" or "Nx,Ny" (e.g., "10,25") |
| `--agg AGG` | Aggregation: mean\|max\|min\|sum\|count |
| `--vmin VMIN` | Color scale min (auto if not set) |
| `--vmax VMAX` | Color scale max (auto if not set) |
| `--cmap CMAP` | Matplotlib colormap |
| `--save SAVE` | Directory to save PNGs (one per frame) |

### `hyst`

#### Examples

- `reaxkit elect hyst --plot --yaxis pol_z --xaxis field_z --aggregate mean --roots`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--xmolout XMOLOUT` | Path to xmolout file |
| `--fort7 FORT7` | Path to fort.7 file |
| `--fort78 FORT78` | Path to fort.78 file |
| `--control CONTROL` | Path to control file |
| `--plot` | Show the hysteresis or time-series plot |
| `--save SAVE` | If set, save plot to file (e.g. hyst.png) |
| `--export EXPORT` | CSV file to export aggregated hysteresis data |
| `--summary SUMMARY` | Text file to write coercive fields and remnant polarizations |
| `--yaxis YAXIS` | Quantity for y-axis (e.g. pol_z, mu_z, time, P_z) |
| `--xaxis XAXIS` | Quantity for x-axis (e.g. field_z, time, iter) |
| `--aggregate {mean,max,min,last}` | Aggregation method |
| `--roots` | Also print coercive and remnant values to terminal |

### `plot3d`

#### Examples

- `reaxkit elect plot3d --core Al --component mu_z --frames 0:3:1 --save reaxkit_outputs/elect/local_mu_3d/`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--xmolout XMOLOUT` | Path to xmolout file |
| `--fort7 FORT7` | Path to fort.7 file |
| `--core CORE` | Comma-separated core atom types (e.g. Al,Mg) |
| `--component COMPONENT` | Which component to color by (e.g. pol_z, P_z (uC/cm^2), mu_z) |
| `--frames FRAMES` | Frames: "0,10,20" or "0:100:5" |
| `--polarization` | Use polarization components (P_x/y/z) instead of dipole |
| `--save SAVE` | Directory to save PNGs (one per frame) |
| `--vmin VMIN` | Color scale min (auto if not set) |
| `--vmax VMAX` | Color scale max (auto if not set) |
| `--size SIZE` | Marker size |
| `--alpha ALPHA` | Marker transparency |
| `--cmap CMAP` | Matplotlib colormap |
| `--elev ELEV` | 3D view elevation |
| `--azim AZIM` | 3D view azimuth |
