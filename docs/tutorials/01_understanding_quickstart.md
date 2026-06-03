# Understanding the Quick Start

In [quickstart.md](../quickstart.md), you ran a working ReaxKit CLI command with minimal setup.
This tutorial explains what happened internally, using the current command style.

We walk through:
- how ReaxKit processes trajectory data,
- what each CLI flag does,
- how workflows, tasks, and engine loaders interact,
- the difference between `frame`, `iter`, and `time`,
- and where outputs are written.

---

## Re-running the Quick Start command

Use this command:

```bash
reaxkit timeseries --field trajectory[1].z --xaxis time --export atom1_z.csv
```

This extracts the z-coordinate trajectory of atom 1 and exports it as CSV.

---

## Explaining the CLI flags

`--field trajectory[1].z`
- `trajectory[...]` means coordinate time series from trajectory data.
- `[1]` selects atom id 1 (1-based indexing).
- `.z` selects only the z component.

`--xaxis time`
- Controls x-axis representation.
- Allowed values are `frame`, `iter`, `time`.
- `time` converts iteration index to physical time using the `control` file when available.

`--export atom1_z.csv`
- Writes the computed table to CSV.
- Relative paths are placed under ReaxKit-managed output folders.

---

## What happens internally

When you run the command, the flow is:

1. The `timeseries` workflow parses CLI arguments and resolves the request type from `--field`.
2. ReaxKit loads required simulation/trajectory data through the runtime engine adapter.
3. The matching analysis task runs (for this command: trajectory coordinate series).
4. The result is returned as a structured table/series object.
5. Presentation/export layers write output and optionally render plots.

---

## `frame` vs `iter` vs `time`

### `frame`
- Sequential snapshot index (0-based).
- Always available.

### `iter`
- Simulation iteration number from source data.
- May skip or restart depending on simulation runs/restarts.

### `time`
- Physical time derived from iteration and timestep metadata.
- Requires enough metadata (typically `control`) for conversion.

Best practice: use `frame` for deterministic indexing, and `iter`/`time` for interpretation and plotting.

---

## Output locations

By default:
- exported analysis tables/plots go under `reaxkit_workspace/analysis/`
- generated input files go under `reaxkit_workspace/inputs/`

Example:
- `reaxkit_workspace/analysis/timeseries/run_20260523_115443_cfc58e/atom1_z.csv`

where `run_20260523_115443_cfc58e` is a unique identifier for this specific command execution.

You can override output file paths via `--export` and `--save`.

---

## Related next steps

- See the next tutorial [02_atom_property_and_video_workflows](02_atom_property_and_video_workflows.md) to learn how to make plots and videos.

---

[Back to Tutorials](index.md)
