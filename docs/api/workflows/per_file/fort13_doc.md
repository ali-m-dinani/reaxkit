# Fort13 Workflow

CLI namespace: `reaxkit fort13 <task> [flags]`

fort.13 error-analysis workflow for ReaxKit.

This workflow provides utilities for reading and analyzing ReaxFF `fort.13` files,
which store force-field training and optimization error metrics as a function
of training epoch.

It supports:
- Extracting total force-field error values versus epoch.
- Visualizing error convergence during force-field optimization.
- Exporting error data to CSV for post-processing or comparison across runs.

The workflow is intended for monitoring ReaxFF parameter optimization and
assessing convergence behavior in training or fitting workflows.

## Available tasks

### `get`

#### Examples

- `reaxkit fort13 get --save total_ff_error_vs_epoch.png`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to fort.13 file |
| `--plot` | Plot error vs epoch |
| `--save SAVE` | Save plot image to path |
| `--export EXPORT` | Export data to CSV file |
