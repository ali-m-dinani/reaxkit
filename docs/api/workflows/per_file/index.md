# Per-file Workflows

This section documents **per-file workflows** in ReaxKit.
Per-file workflows operate on **a single ReaxFF output file** and expose
file-specific analyses through the CLI, similar to [per-file analyzers](../../analysis/per_file/index.md).

They are the most direct way to extract information from individual ReaxFF files.

---

## What makes a workflow “per-file”

A per-file workflow:

- Uses **one ReaxFF file type**
- Relies on a **single handler**
- Exposes analyses tightly coupled to that file’s contents
- Maps cleanly to commands of the form:

```bash
reaxkit <file-kind> <task> [options]
```

Examples:
- `reaxkit xmolout get`
- `reaxkit fort7 get`
- `reaxkit fort74 plot`

---

## Typical per-file workflows

Common per-file workflow families include:

- **Trajectory files** (`xmolout`, `vels`)  
  Coordinates, velocities, MSDs, basic time-series analysis.

- **Energetics and logs** (`energylog`, `fort.73`, `fort.58`)  
  Energies and related quantities vs iteration or time.

- **Scalar outputs** (`fort.74`, `fort.76`)  
  Volume, density, pressure, stress, and similar properties.

- **Connectivity and bonding** (`fort.7`, `fort.57`)  
  Bond orders, coordination numbers, and connectivity summaries.

- **Force-field inputs** (`ffield`, `params`)  
  Parameter inspection and interpretation.

---

## When to use per-file workflows

Use per-file workflows when:

- Your analysis depends on **only one file**
- You want fast inspection or plotting
- You are exploring raw simulation outputs

For analyses that require multiple files or structural context,
see **composed workflows** instead.

---

Per-file workflows form the **core analysis interface** of ReaxKit,
bridging raw ReaxFF files to usable data with minimal overhead.
