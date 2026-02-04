# Summary Workflow

CLI namespace: `reaxkit summary <task> [flags]`

summary.txt analysis workflow for ReaxKit.

This workflow provides tools for reading, analyzing, and visualizing data from
ReaxFF `summary.txt` files, which contain per-iteration thermodynamic and
simulation summary quantities.

It supports:
- Extracting a selected summary column (with alias support) as a function of
  iteration, frame index, or physical time.
- Converting the x-axis between iteration, frame, and time using control-file
  metadata.
- Selecting subsets of frames for focused analysis.
- Plotting summary quantities, saving figures, or exporting the processed data
  to CSV using standardized output paths.

The workflow is designed for quick inspection and post-processing of ReaxFF
summary outputs, enabling reproducible analysis of thermodynamic and
simulation-wide properties.

## Available tasks

### `get`

#### Examples

- `reaxkit summary get --yaxis E_pot --xaxis time --plot`
- `reaxkit summary get --file summary.txt --yaxis T --xaxis iter --frames 0:400:5 --save summary_T_vs_iter.png --export summary_T_vs_iter.csv`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to summary file |
| `--xaxis {time,iter,frame}` | X-axis domain (default: time) |
| `--yaxis YAXIS` | Y-axis feature/column (aliases allowed, e.g., 'E_potential' → 'E_pot') |
| `--frames FRAMES` | Frames to select: 'start:stop[:step]' or 'i,j,k' (default: all) |
| `--plot` | Show the plot interactively. |
| `--save SAVE` | Save the plot to a file (without showing). Provide a path. |
| `--export EXPORT` | Export the data to CSV. Provide a path. |
