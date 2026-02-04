# Ffield Workflow

CLI namespace: `reaxkit ffield <task> [flags]`

Force-field (ffield) workflow for ReaxKit.

This workflow provides tools for inspecting, filtering, interpreting, and
exporting data from ReaxFF `ffield` files, which define the full set of force-field
parameters (atoms, bonds, angles, torsions, off-diagonals, and hydrogen bonds).

It supports:
- Extracting individual ffield sections in either interpreted (symbol-based)
  or index-based form.
- Filtering term-based parameters using flexible syntax (e.g. C-H, CCH, 1-2,
  with ordered or any-order matching where appropriate).
- Exporting selected sections or the entire force field to CSV for analysis,
  comparison, or parameterization workflows.

The workflow is designed to make force-field inspection transparent and
scriptable, supporting both exploratory analysis and large-scale ReaxFF
parameter studies.

## Available tasks

### `export`

#### Examples

- `# Export everything interpreted (C-H, C-C-H, ...)`
- `reaxkit ffield export --format interpreted --outdir reaxkit_outputs/ffield/all_ffield_csv`
- `# Export everything in indices (1-2, 1-1-2, ...)`
- `reaxkit ffield export --format indices --outdir reaxkit_outputs/ffield/all_ffield_csv`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to ffield file. |
| `--format {interpreted,indices}` | Export format: interpreted uses atom symbols, indices uses numeric atom indices. |
| `--outdir OUTDIR` | Directory to write CSVs (will be placed under reaxkit_output/...). |

### `get`

#### Examples

- `# 1) Get all C-H bond rows (interpreted output)`
- `reaxkit ffield get --section bond --term C-H --format interpreted --export CH_bond.csv`
- `# 2) Same, but using indices`
- `reaxkit ffield get --section bond --term 1-2 --format indices --export 1_2_bond.csv`
- `# 3) Angles: get only C-C-H (ordered)`
- `reaxkit ffield get --section angle --term CCH --format interpreted --export CCH_angles.csv`
- `# 4) Angles: get all combinations of C-C-H in angle data`
- `reaxkit ffield get --section angle --term CCH --format interpreted --any-order --export all_CCH_angles.csv`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to ffield file. |
| `--format {interpreted,indices}` | Output format: interpreted uses atom symbols (C-H), indices uses numeric (1-2). |
| `--export EXPORT` | Path to export CSV. If omitted, prints a preview. |
| `--section SECTION` | Section: general, atom, bond, off_diagonal, angle, torsion, hbond. |
| `--term TERM` | Filter term, e.g. 'C-H', 'CCH', 'C-C-H', '1-2', '1-1-2'. |
| `--ordered-2body` | For 2-body sections (bond/off_diagonal), treat (C-H) and (H-C) as different. Default is unordered. |
| `--any-order` | Match all permutations of the given term (e.g. CCH matches CCH, CHC, HCC). |
