# Fort78 Workflow

CLI namespace: `reaxkit fort78 <task> [flags]`

fort.78 electric-field output workflow for ReaxKit.

This workflow provides tools for reading, visualizing, and exporting data from
ReaxFF `fort.78` files, which record time-dependent electric-field–related
quantities during simulations with applied external fields.

It supports:
- Extracting a single electric-field or related scalar component (e.g.
  `E_field_x`) as a function of iteration, frame index, or physical time.
- Converting the x-axis between iteration, frame, and time using the associated
  control file when required.
- Plotting the selected quantity, saving figures to disk, or exporting the data
  to CSV using standardized output paths.

The workflow is intended for analysis of electric-field protocols, field-driven
responses, and post-processing of ReaxFF simulations involving external fields.

## Available tasks

### `get`

#### Examples

- `reaxkit fort78 get --xaxis time --yaxis E_field_x --save E_field_x.png --export E_field_x.csv`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to fort.78 file |
| `--yaxis YAXIS` | Name of the fort.78 yaxis to extract (e.g., 'E_field_x') |
| `--xaxis {iter,frame,time}` | X-axis for plotting/export (default: iter). 'time' may require a control file. |
| `--control CONTROL` | Path to control file (only used when --xaxis time). |
| `--plot` | Show a plot in a window |
| `--save SAVE` | Save plot to a path or directory |
| `--export EXPORT` | Export data to CSV at this path |
