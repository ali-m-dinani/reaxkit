# Coordination Workflow

CLI namespace: `reaxkit coordination <task> [flags]`

Coordination workflow for ReaxKit.

This workflow analyzes atomic coordination using ReaxFF bond-order data
from `fort.7` and structural information from `xmolout`.

It provides two main capabilities:

- Analyze coordination status per atom per frame (under-, correctly-, or over-coordinated)
  based on atom valence and total bond order.
- Relabel atom types according to coordination status and write a new `xmolout` file
  for visualization or downstream analysis.

The workflow supports flexible frame selection, user-defined or ffield-derived valences,
and multiple labeling modes.

## Available tasks

### `analyze`

Analyzes coordination per atom per frame.<br>
It determines if an atom is under-coordinated, over-coordinated, or coordinates, based on its valence and total bond order.

#### Examples

- `reaxkit coord analyze --export coord_analysis.csv`
- `reaxkit coord analyze --valences 'Mg=2,O=2' --frames 0:200:2 --export coord_0_200_2.csv`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--xmolout XMOLOUT` | Path to xmolout. |
| `--fort7 FORT7` | Path to fort.7 file. |
| `--valences VALENCES` | Optional: override type valences, e.g. 'Mg=2,O=2,H=1'. If omitted, reads from ffield atom section ('valency'). |
| `--ffield FFIELD` | Path to ffield (used if --valences not given). |
| `--threshold THRESHOLD` | Tolerance around valence. |
| `--frames FRAMES` | Frame selection, e.g. '0:100:5' or '0,5,10'. |
| `--allow-missing-valences` | Keep atoms with unknown valence (status=NaN). |
| `--export EXPORT` | Path to export coordination CSV. |

### `relabel`

Relabel atom types by coordination and write a new xmolout.

#### Examples

- `reaxkit coord relabel --output xmolout_relabeled --mode global --labels=-1=U,0=C,1=O`
- `reaxkit coord relabel --output xmolout_type --mode by_type --keep-coord-original`
- `reaxkit coord relabel --valences 'Mg=2,O=2,Zn=2' --frames 0:400:5 --output xmolout_relabeled --mode global`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--xmolout XMOLOUT` | Path to xmolout. |
| `--fort7 FORT7` | Path to fort.7 file. |
| `--valences VALENCES` | Optional: override type valences, e.g. 'Mg=2,O=2,H=1'. If omitted, reads from ffield atom section ('valency'). |
| `--ffield FFIELD` | Path to ffield (used if --valences not given). |
| `--threshold THRESHOLD` | Tolerance around valence. |
| `--frames FRAMES` | Frame selection, e.g. '0:100:5' or '0,5,10'. |
| `--allow-missing-valences` | Keep atoms with unknown valence (status=NaN). |
| `--output OUTPUT` | Output xmolout path. |
| `--mode {global,by_type}` | Relabeling mode. |
| `--labels LABELS` | Status→tag map, e.g. '-1=U,0=C,1=O'. |
| `--keep-coord-original` | In by_type mode, keep original label when status==0. |
| `--simulation SIMULATION` | Override header simulation name. |
| `--precision PRECISION` | Float precision. |
