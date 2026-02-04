# Fort76 Workflow

CLI namespace: `reaxkit fort76 <task> [flags]`

fort.76 restraint-analysis workflow for ReaxKit.

This workflow provides tools for reading, visualizing, and exporting data from
ReaxFF `fort.76` files, which record restraint targets and actual values during
MD or minimization runs.

It supports:
- Extracting and plotting a single restraint-related column versus iteration,
  frame index, or physical time.
- Comparing restraint target and actual values for a selected restraint index
  as a function of iteration, frame, or time.
- Converting the x-axis between iteration, frame, and time using the associated
  control file.
- Saving plots to disk or exporting data to CSV using standardized output paths.

The workflow is intended for diagnosing restraint behavior, convergence, and
stability in constrained ReaxFF simulations.

## Available tasks

### `get`

#### Examples

- `reaxkit fort76 get --ycol E_res --xaxis time --save E_res_vs_time.png`
- `reaxkit fort76 get --ycol r1_actual --xaxis frame --export r1_actual_vs_frame.csv`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to fort.76 file. |
| `--xaxis {iter,frame,time}` |  |
| `--control CONTROL` | Needed for --xaxis time. |
| `--plot` | Show plot interactively. |
| `--save SAVE` | Path to save plot image. |
| `--export EXPORT` | Path to export CSV data. |
| `--title TITLE` |  |
| `--xlabel XLABEL` |  |
| `--ylabel YLABEL` |  |
| `--ycol YCOL` | Column to plot/export (aliases allowed). |

### `respair`

#### Examples

- `reaxkit fort76 respair --restraint 2 --xaxis time --control control --save r2_vs_time.png`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to fort.76 file. |
| `--xaxis {iter,frame,time}` |  |
| `--control CONTROL` | Needed for --xaxis time. |
| `--plot` | Show plot interactively. |
| `--save SAVE` | Path to save plot image. |
| `--export EXPORT` | Path to export CSV data. |
| `--title TITLE` |  |
| `--xlabel XLABEL` |  |
| `--ylabel YLABEL` |  |
| `--restraint RESTRAINT` | Restraint index (1-based). |
