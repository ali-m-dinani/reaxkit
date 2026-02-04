# Eregime Workflow

CLI namespace: `reaxkit eregime <task> [flags]`

Electric-field regime (eregime) workflow for ReaxKit.

This workflow provides tools to read, analyze, visualize, and generate
`eregime.in` files, which define time- or iteration-dependent electric-field
schedules in ReaxFF simulations.

It supports:
- Extracting electric-field components (e.g. E1, E2) versus iteration, frame,
  or physical time, with optional plotting and CSV export.
- Selecting subsets of data by frame ranges for focused analysis.
- Generating new `eregime.in` files using standard field profiles, including
  sinusoidal waves, smooth pulses, or user-defined analytic functions.

The workflow is designed to bridge ReaxFF electric-field protocols with
analysis and visualization, enabling reproducible setup and interpretation
of field-driven simulations.

## Available tasks

### `gen`

#### Examples

- `reaxkit eregime gen sin --output eregime_sin.in --max-magnitude 0.004 --step-angle 0.05 --iteration-step 500 --num-cycles 2 --direction z --V 1`
- `reaxkit eregime gen pulse --output eregime_pulse.in --amplitude 0.003 --width 50 --period 200 --slope 20 --iteration-step 250 --num-cycles 5 --direction z --V 1`
- `reaxkit eregime gen func --output eregime_func.in --expr '0.003*cos(2*pi*t/100)' --t-end 1000 --dt 1 --iteration-step 250 --direction z --V 1`
- `positional arguments:`
- `{sin,pulse,func}`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |

### `get`

#### Examples

- `reaxkit eregime get --column E1 --xaxis time --export eregime_E1_vs_time.csv --save eregime_E1_vs_time.png`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to eregime.in file |
| `--column COLUMN` | Y column to extract (aliases supported, e.g., E, E1, E2, direction, direction1, direction2) |
| `--xaxis {iter,frame,time}` | X axis for the output: 'iter' (default), 'frame', or 'time' (needs --control) |
| `--control CONTROL` | Control file used to convert iteration → time when --xaxis time (default: control) |
| `--frames FRAMES` | Row selector (position-based): 'start:stop[:step]' or 'i,j,k' (default: all rows) |
| `--plot` | Show the plot interactively. |
| `--save SAVE` | Save the plot image to this path. |
| `--export EXPORT` | Export data CSV to this path. |
