# Control Workflow

CLI namespace: `reaxkit control <task> [flags]`

Control-file workflow for ReaxKit.

This workflow provides utilities for inspecting and generating ReaxFF
`control` files, which define global simulation settings such as
MD/energy-minimization parameters, output frequencies, and algorithmic options.

It supports:
- Querying individual control keys (e.g. `nmdit`, `iout2`) from an existing
  control file, with optional section filtering.
- Generating a complete default control file containing all standard sections
  and recommended default values.

The workflow is designed to make control-file management transparent,
scriptable, and reproducible from the command line.

## Available tasks

### `get`

#### Examples

- `reaxkit control get nmdit`
- `positional arguments:`
- `key                Control key to look up, e.g. 'nmdit'.`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to control file (default: 'control'). |
| `--section SECTION` | Optional control section (general, md, mm, ff, outdated). |

### `make`

#### Examples

- `reaxkit control make`
- `reaxkit control make --output control`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--output OUTPUT` | Output path for the generated control file (default: 'control'). |
