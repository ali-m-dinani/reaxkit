# Vregime Workflow

CLI namespace: `reaxkit vregime <task> [flags]`

Volume-regime (vregime) workflow for ReaxKit.

This workflow provides a utility for generating sample `vregime.in` files,
which define volume or pressure control schedules used in ReaxFF simulations.

It supports:
- Writing a correctly formatted `vregime.in` file with fixed-width columns.
- Controlling the number of volume-regime rows written to the file.
- Automatically selecting a standardized output location when no explicit
  output path is provided.

The workflow is intended to simplify creation of valid volume-regime input
files for testing, prototyping, and reproducible ReaxFF simulation setup.

## Available tasks

### `gen`

#### Examples

- `reaxkit vregime gen`
- `reaxkit vregime gen --rows 3`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--out OUT` | Output vregime filename or path. If not provided, writes to reaxkit_outputs/vregime/vregime.in |
| `--rows ROWS` | Number of sample rows to write. |
