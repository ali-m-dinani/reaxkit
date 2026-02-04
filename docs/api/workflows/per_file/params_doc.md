# Params Workflow

CLI namespace: `reaxkit params <task> [flags]`

Params-file workflow for ReaxKit.

This workflow provides tools for inspecting and exporting ReaxFF `params` files,
which define optimization parameters and search intervals used during
force-field training.

It supports:
- Loading the raw params table with optional duplicate removal and flexible
  column-based sorting.
- Interpreting params entries by resolving their references into the
  corresponding sections and rows of the `ffield` file.
- Optionally constructing human-readable chemical terms (e.g. C–C–H) during
  interpretation for improved readability.
- Exporting processed params data to CSV for downstream analysis, auditing,
  or force-field development workflows.

The workflow is designed to bridge low-level optimization parameter definitions
with interpretable force-field context in a reproducible, CLI-driven manner.

## Available tasks

### `get`

#### Examples

- `reaxkit params get --export params.csv`
- `Interpreted params:`
- `reaxkit params get --interpret --export params_interpreted.csv`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to params file. |
| `--export EXPORT` | Path to export CSV data. |
| `--keep-duplicates` | If set, do NOT drop duplicates (default drops duplicates). |
| `--sort-by SORT_BY` | Optional column name to sort by (default: no sorting). |
| `--descending` | If set, sort in descending order (only if --sort-by is used). |
| `--interpret` | If set, interpret params pointers into the ffield (adds section/row/param/value/term columns). |
| `--ffield FFIELD` | Path to ffield file (required when --interpret is set). |
| `--no-term` | If set, do not build readable term (e.g., C-C-H) during interpretation. |
