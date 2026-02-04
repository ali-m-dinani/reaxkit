# Xmolout Workflow

CLI namespace: `reaxkit xmolout <task> [flags]`

xmolout trajectory-analysis workflow for ReaxKit.

This workflow provides a comprehensive set of tools for extracting, analyzing,
and visualizing atomic trajectory data from ReaxFF `xmolout` files, which store
time-resolved atomic coordinates and simulation cell information.

It supports:
- Extracting atom trajectories for selected atoms or atom types in long or wide
  tabular formats, with flexible frame selection and axis conversion.
- Computing mean squared displacement (MSD) for one or multiple atoms, with
  support for combined plots or per-atom subplots.
- Computing radial distribution functions (RDFs) using FREUD or OVITO backends,
  either as averaged curves or as RDF-derived properties over frames.
- Extracting simulation box (cell) dimensions as functions of frame, iteration,
  or time, with optional plotting.
- Writing trimmed `xmolout` files that retain only atom types and coordinates
  for lightweight storage or downstream processing.

The workflow is designed to bridge raw ReaxFF trajectory output with common
structural and dynamical analyses in a reproducible, CLI-driven manner.

## Available tasks

### `boxdims`

Get box/cell dimensions from xmolout.

#### Examples

- `reaxkit xmolout boxdims --frames 0:500:5 --xaxis time --export box_dims.csv`
- `reaxkit xmolout boxdims --frames 0:100:10 --xaxis iter --save box_dim_plots.png`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to xmolout file. |
| `--plot` | Show plot interactively. |
| `--save SAVE` | Path to save plot image. |
| `--export EXPORT` | Path to export CSV data. |
| `--frames FRAMES` | Frame selector: 'start:stop[:step]' or 'i,j,k'. |
| `--xaxis {frame,iter,time}` | Quantity on x-axis (default: frame). |

### `msd`

Compute MSD from xmolout.

#### Examples

- `reaxkit xmolout msd --atoms 1 --xaxis frame --save atom1_msd.png --export atom1_msd.csv`
- `reaxkit xmolout msd --atoms 1,2,3 --xaxis time --save msd.png --export msd.csv`
- `reaxkit xmolout msd --atoms 1,2,3 --subplot --save msd_subplot.png --export msd_subplot.csv`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to xmolout file. |
| `--plot` | Show plot interactively. |
| `--save SAVE` | Path to save plot image. |
| `--export EXPORT` | Path to export CSV data. |
| `--atoms ATOMS` | Comma/space separated 1-based atom indices, e.g. '1,5,12'. |
| `--xaxis {iter,frame,time}` | Quantity on x-axis (default: frame). |
| `--subplot` | Plot each atom in its own subplot instead of a single combined plot. |

### `rdf`

Compute RDF using FREUD or OVITO backends.<br>
<br>
Curve example:<br>
  reaxkit xmolout rdf --save rdf.png --export rdf.csv --frames 0 --bins 200 --r-max 5<br>
<br>
Property example:<br>
  reaxkit xmolout rdf --prop area --bins 200 --r-max 5 --frames 0:10:1 --save rdf_area.png

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to xmolout file. |
| `--plot` | Show plot interactively. |
| `--save SAVE` | Path to save plot image. |
| `--export EXPORT` | Path to export CSV data. |
| `--backend {freud,ovito}` | RDF backend: freud or ovito (default: ovito). |
| `--prop {first_peak,dominant_peak,area,excess_area}` | Compute this RDF-derived property per frame instead of a curve. |
| `--types-a TYPES_A, --types_a TYPES_A` | Comma/space separated atom types for set A, e.g. 'Al,N'. |
| `--types-b TYPES_B, --types_b TYPES_B` | Comma/space separated atom types for set B, e.g. 'N'. |
| `--bins BINS` | Number of RDF bins. |
| `--r-max R_MAX` | Max radius in Å; default depends on backend. |
| `--frames FRAMES` | Frame selector: 'start:stop[:step]' or 'i,j,k'. |
| `--every EVERY` | Use every Nth frame (default: 1). |
| `--start START` | First frame index (0-based). |
| `--stop STOP` | Last frame index (0-based). |
| `--norm {extent,cell}` | FREUD normalization: extent or cell (default: extent). |
| `--c-eff C_EFF` | FREUD only: effective c-length for --norm cell. |

### `trajget`

Get atom trajectories from xmolout.

#### Examples

- `reaxkit xmolout trajget --atoms 1 --dims z --xaxis time --save atom1_z.png --export atom1_z.csv`
- `reaxkit xmolout trajget --atom-types Al --dims x y z --format wide --export Al_all_dims_traj.csv`
- `reaxkit xmolout trajget --atom-types Al --frames 10:200:10 --dims z --export Al_z_dim_traj.csv`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to xmolout file. |
| `--plot` | Show plot interactively. |
| `--save SAVE` | Path to save plot image. |
| `--export EXPORT` | Path to export CSV data. |
| `--atoms ATOMS` | Comma/space separated 1-based atom indices, e.g. '1,5,12'. |
| `--dims {x,y,z} [{x,y,z} ...]` | Coordinate dimensions to include (default: x y z). |
| `--xaxis {iter,frame,time}` | Quantity on x-axis (default: frame). |
| `--frames FRAMES` | Frame selector: 'start:stop[:step]' or 'i,j,k' (default: all). |
| `--atom-types ATOM_TYPES, --types ATOM_TYPES` | Comma/space separated atom types, e.g. 'Al,N'. |
| `--format {long,wide}` | Output table layout: long or wide (default: long). |

### `trim`

Trim xmolout to a lighter file with atom type and coordinates only.<br>
<br>
Example:<br>
  reaxkit xmolout trim --file xmolout --output xmolout_trimmed

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Input xmolout file. |
| `--output OUTPUT` | Output trimmed xmolout file. |
