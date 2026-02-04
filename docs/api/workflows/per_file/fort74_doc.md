# Fort74 Workflow

CLI namespace: `reaxkit fort74 <task> [flags]`

fort.74 summary-table workflow for ReaxKit.

This workflow provides utilities for reading and exporting data from ReaxFF
`fort.74` files, which contain per-structure or per-configuration summary
quantities produced during force-field training or evaluation runs.

It supports:
- Exporting all available columns from a `fort.74` file to CSV.
- Selecting and exporting a single column (with optional alias resolution,
  e.g. `Density` → `D`) while always preserving the structure identifier.
- Writing outputs to a standardized ReaxKit output directory for
  reproducible data organization.

The workflow is designed for lightweight inspection and downstream analysis
of ReaxFF summary metrics such as energies, volumes, and densities.

## Available tasks

### `get`

#### Examples

- `reaxkit fort74.md get --export fort74_all_data.csv`
- `reaxkit fort74.md get --col Emin --export fort74_all_Emin_data.csv`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to fort.74 file. |
| `--col COL` | Column to export, or 'all'. |
| `--export EXPORT` | CSV output path. If a bare filename is given, it will be saved under reaxkit_outputs/fort74.md/ (via path.resolve_output_path). |
