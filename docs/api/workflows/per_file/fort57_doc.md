# Fort57 Workflow

CLI namespace: `reaxkit fort57 <task> [flags]`

fort.57 thermodynamic-output workflow for ReaxKit.

This workflow provides tools for reading, visualizing, and exporting data from
ReaxFF `fort.57` files, which typically contain thermodynamic and simulation
monitoring quantities recorded during MD runs.

It supports:
- Extracting one or more scalar quantities (e.g. potential energy, temperature,
  RMS force, number of force calls) as functions of iteration, frame, or time.
- Converting the x-axis between iteration, frame index, and physical time using
  the associated control file.
- Plotting selected quantities or exporting them to CSV for downstream analysis.

The workflow is intended for quick inspection and post-processing of ReaxFF
simulation stability, convergence, and thermodynamic behavior.

## Available tasks

### `get`

#### Examples

- `reaxkit fort57.md get --yaxis RMSG --xaxis iter --save rmsg_vs_iter.png`
- `reaxkit fort57.md get --yaxis all --xaxis iter --export fort57_all.csv`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to fort.57 file. |
| `--control CONTROL` | Path to control file (for --xaxis time). |
| `--xaxis {iter,frame,time}` | X-axis: iter, frame, or time (time uses control:tstep). |
| `--yaxis YAXIS` | Y column(s): e.g. 'RMSG' or 'iter RMSG' or 'E_pot,T' or 'all'. |
| `--plot` | Show plot interactively. |
| `--save SAVE` | Path to save plot image (suffix _<col> if multiple y). |
| `--export EXPORT` | Path to export CSV (x + selected y columns). |
