# Molfra Workflow

CLI namespace: `reaxkit molfra <task> [flags]`

Molecular-fragment (molfra) analysis workflow for ReaxKit.

This workflow provides tools for analyzing ReaxFF `molfra.out` and
`molfra_ig.out` files, which describe molecular fragments, species
identities, and their evolution during a simulation.

It supports:
- Tracking occurrences of selected molecular species across frames,
  with optional automatic selection based on occurrence thresholds.
- Computing and visualizing global totals, including total number of
  molecules, total atoms, and total molecular mass versus iteration,
  frame, or physical time.
- Identifying and analyzing the largest molecule in the system, either
  by individual molecular mass or by per-element atom composition.
- Plotting results, saving figures, and exporting processed data to CSV
  using standardized ReaxKit output paths.

The workflow is designed to enable systematic analysis of chemical
speciation, fragmentation, and growth processes in ReaxFF simulations.

## Available tasks

### `largest`

#### Examples

- `reaxkit molfra largest --atoms --frames '0:30:2' --xaxis time --save largest.png --export largest.csv`
- `reaxkit molfra largest --mass --xaxis time --export largest_mass.csv`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to molfra.out |
| `--atoms` | Use per-element atom counts for the largest molecule per iter (default). |
| `--mass` | Use largest molecule individual mass vs x-axis. |
| `--xaxis XAXIS` | X-axis mode. Canonical: 'iter', 'time', 'frame'. |
| `--control CONTROL` | Path to control file for time conversion (used when --xaxis time). |
| `--frames FRAMES` | Frame selection (position-based after filtering): 'start:stop[:step]' or 'i,j,k'. |
| `--title TITLE` | Custom plot title. |
| `--plot` | Show the plot interactively. |
| `--save SAVE` | Save the plot (path or directory, resolved via resolve_output_path). |
| `--export EXPORT` | Export the data table to CSV (path or directory, resolved via resolve_output_path). |

### `occur`

#### Examples

- `reaxkit molfra occur --molecules H2O N128Al128 --save water_and_slab_occurrence.png`
- `reaxkit molfra occur --threshold 3 --exclude Pt --export species_with_max_occur.csv`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to molfra.out |
| `--molecules MOLECULES [MOLECULES ...]` | One or more molecule types to include (e.g., H2O OH N128Al128). |
| `--threshold THRESHOLD` | Auto-include all species whose max occurrence ≥ threshold. |
| `--exclude [EXCLUDE ...]` | Species to exclude when using --threshold (e.g., Pt). |
| `--xaxis XAXIS` | X-axis mode. Canonical: 'iter', 'time', 'frame'. |
| `--control CONTROL` | Path to control file for time conversion (used when --xaxis time). |
| `--frames FRAMES` | Frame selection (position-based after filtering): 'start:stop[:step]' or 'i,j,k'. |
| `--title TITLE` | Custom plot title. |
| `--plot` | Show the plot interactively. |
| `--save SAVE` | Save the plot (path or directory, resolved via resolve_output_path). |
| `--export EXPORT` | Export the data table to CSV (path or directory, resolved via resolve_output_path). |

### `total`

#### Examples

- `reaxkit molfra total --file molfra_ig.out --export totals_data.csv --save totals_data.png`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to molfra.out |
| `--xaxis XAXIS` | X-axis mode. Canonical: 'iter', 'time', 'frame'. |
| `--control CONTROL` | Path to control file for time conversion (used when --xaxis time). |
| `--frames FRAMES` | Frame selection (position-based after filtering): 'start:stop[:step]' or 'i,j,k'. |
| `--title TITLE` | Custom plot title. |
| `--plot` | Show the multi-subplot figure interactively. |
| `--save SAVE` | Save the multi-subplot figure (path or directory, resolved via resolve_output_path). |
| `--export EXPORT` | Export all totals to CSV (path or directory, resolved via resolve_output_path). |
