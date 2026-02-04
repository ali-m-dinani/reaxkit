# Fort73 Workflow

CLI namespace: `reaxkit fort73 <task> [flags]`

Energy-output workflow for ReaxKit (fort.73 / energylog / fort.58).

This workflow provides tools for reading, visualizing, and exporting energy
and thermodynamic data produced by ReaxFF simulations, as stored in
`fort.73`, `energylog`, or `fort.58` files.

It supports:
- Extracting individual or all available energy terms (e.g. Ebond, Eangle,
  Etot) as functions of iteration, frame index, or physical time.
- Converting the x-axis between iteration, frame, and time using the
  associated control file.
- Plotting selected energy components or saving them as image files.
- Exporting energy data to CSV for downstream analysis or comparison
  across simulation runs.

The workflow is designed for rapid inspection of ReaxFF energy evolution,
stability, and convergence behavior.

## Available tasks

### `get`

#### Examples

- `reaxkit fort73 get --yaxis Ebond --xaxis time --plot`
- `reaxkit fort73 get --yaxis all --xaxis time --save reaxkit_outputs/fort73/`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to fort.73 / energylog file |
| `--yaxis YAXIS` | Energy column (e.g. Ebond) or 'all' |
| `--xaxis {iter,frame,time}` | X-axis type |
| `--control CONTROL` | Control file (used when xaxis=time) |
| `--export EXPORT` | Path to export CSV (x + selected y columns) |
| `--save SAVE` | Path to save plot image (suffix _<col> if yaxis=all) |
| `--plot` | If set, generate plot(s). |
