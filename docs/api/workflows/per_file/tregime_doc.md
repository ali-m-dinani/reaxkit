# Tregime Workflow

CLI namespace: `reaxkit tregime <task> [flags]`

Temperature-regime (tregime) workflow for ReaxKit.

This workflow provides a utility for generating sample `tregime.in` files,
which define temperature schedules used in ReaxFF molecular dynamics
simulations.

It supports:
- Writing a correctly formatted `tregime.in` file with fixed-width columns.
- Controlling the number of temperature-regime rows written to the file.
- Automatically selecting a standardized output location when no explicit
  output path is provided.

The workflow is intended to simplify creation of valid temperature-regime
input files for testing, prototyping, and reproducible simulation setup.

## Available tasks

### `gen`

#### Examples

- `reaxkit tregime gen`
- `reaxkit tregime gen --rows 5`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--out OUT` | Output tregime filename or path. If not provided, writes to reaxkit_outputs/tregime/tregime.in |
| `--rows ROWS` | Number of sample rows to write. |
