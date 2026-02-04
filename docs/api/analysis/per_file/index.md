# Per-file Analyses

This section documents **analyses that operate on a single ReaxFF output file**.
Each page describes what can be computed *using only one handler* without requiring
data from other files.

These analyses are the building blocks of ReaxKit; composed analyses build on top
of them.

---

## What “per-file” means in ReaxKit

A *per-file analysis*:

- Uses **exactly one ReaxFF file type**
- Depends on **one handler** (`*_handler.py`)
- Produces derived quantities, plots, or summaries from that file alone
- Is typically exposed via a CLI command of the form:

```bash
reaxkit <file-kind> <task> ...
```

Examples:
- `reaxkit xmolout get`
- `reaxkit fort74 get`
- `reaxkit energylog plot`

---

## Available per-file analysis groups

Each subpage below corresponds to a ReaxFF file type and its supported analyses. To better understand the following categories, we suggest looking at [ReaxFF documentation](../../../reaxff_reference/index.md).

### Trajectory / structure files

- [xmolout](xmolout_analyzer_doc.md)  
  Atomic coordinates, frames, trajectories, MSDs, basic geometry analysis.

- [vels](vels_analyzer_doc.md)  
  Atomic velocities and velocity-derived statistics.

---

### Energetics / thermodynamics

- [energylog / fort.73 / fort.58](fort73_analyzer_doc.md)  
  Energies vs iteration/time (total, bonded, non-bonded, electrostatic, etc.).

- [fort.76](fort76_analyzer_doc.md)  
  Stress, pressure, and related quantities.

- [fort.74](fort74_analyzer_doc.md) 
  Scalar properties per iteration (e.g. volume, density, enthalpy).

- [fort.99](fort99_analyzer_doc.md)  
  Training / optimization targets and error metrics.

---

### Connectivity / bonding

- [fort.7](fort7_analyzer_doc.md)  
  Bond orders, connectivity tables, coordination numbers.

- [fort.57](fort57_analyzer_doc.md)
  Bond statistics and related summaries.

---

### Force-field and input-related files

- [ffield](ffield_analyzer_doc.md)  
  Force-field parameters (atoms, bonds, angles, torsions, h-bonds).

- [params](params_analyzer_doc.md)  
  Optimization parameters and bounds.

---

## What each per-file page contains

Each file-specific page typically documents:

1. **Supported analyses** (what can be computed)
2. **Required handler**
3. **Expected DataFrame schema**
4. **Python API examples**
5. **CLI usage examples**
6. **Aliases and column naming rules**
7. **Notes on units, iterations, and frames**

---

## When to look at composed analyses instead

If you need to:

- Combine **coordinates + connectivity**
- Compute **local or cluster-level properties**
- Derive quantities that depend on **multiple files**

→ See the [composed analyses](../composed/index.md) section instead.
