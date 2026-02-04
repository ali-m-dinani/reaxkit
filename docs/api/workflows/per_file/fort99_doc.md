# Fort99 Workflow

CLI namespace: `reaxkit fort99 <task> [flags]`

fort.99 force-field training analysis workflow for ReaxKit.

This workflow provides tools for analyzing ReaxFF `fort.99` files, which contain
energy-error data used during force-field training and validation.

It supports:
- Extracting and sorting ENERGY error tables from `fort.99` files, with optional
  CSV export for downstream analysis.
- Constructing energy–volume (EOS) relationships by combining `fort.99` and
  `fort.74` data, enabling comparison between force-field and QM reference energies.
- Visualizing EOS curves and exporting the underlying data.
- Computing bulk moduli using a Vinet equation-of-state fit, based on selected
  energy identifiers and data sources.

The workflow is intended to support force-field development, validation, and
benchmarking by bridging ReaxFF training outputs with physically meaningful
material properties.

## Available tasks

### `bulk`

#### Examples

- `reaxkit fort99 bulk --iden 'Al2N2_w_opt2'`
- `reaxkit fort99 bulk --iden Al2N2_w_opt2 --source qm`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--fort99 FORT99` | fort.99 file to use |
| `--fort74.md FORT74.MD` | fort.74 file to use |
| `--iden IDEN` | ENERGY identifier (iden1) to use for EOS fitting |
| `--source {ffield,qm}` | Energy source to use (ffield or qm) |

### `eos`

#### Examples

- `reaxkit fort99 eos --iden all --save reaxkit_outputs/fort99/eos_plots/ --flip --export eos_plots.csv`
- `reaxkit fort99 eos --iden bulk_0 --save eos_bulk_0.png --export eos_bulk0.csv`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--fort99 FORT99` | Path to fort.99 file |
| `--fort74.md FORT74.MD` | Path to fort.74 (default: same directory as fort.99) |
| `--iden IDEN` | iden1 to include ('all' or specific e.g. bulk_0) |
| `--plot` | Show plots interactively |
| `--save SAVE` | Directory to save plots as <iden1>.png |
| `--export EXPORT` | CSV file to export ENERGY vs volume table |
| `--flip` | Flip the sign of both QM and force-field energies before plotting |

### `get`

#### Examples

- `reaxkit fort99 get --sort error --ascending --export fort99_sorted_data.csv`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to fort.99 file |
| `--sort SORT` | Column to sort by (e.g., error, ffield_value, qm_value) |
| `--ascending` | Sort ascending (default: descending) |
| `--export EXPORT` | CSV file to export the sorted table (optional) |
