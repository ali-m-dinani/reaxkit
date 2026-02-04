# xmolout + fort.7: Multi-file Coordination and Property Analysis

In the previous tutorial, we analyzed a single ReaxFF output file (`xmolout`)
and learned how ReaxKit processes trajectories using handlers, analyzers, and
workflows.

In this tutorial, we move to the **first multi-file workflow**:
combining **geometric information** from `xmolout` with **bond- and atom-level
properties** from `fort.7`.

This pattern is extremely common in ReaxFF analysis and underpins workflows such
as:
- coordination analysis,
- spatial charge distribution,
- bond-order–resolved visualization,
- structure–property correlations.

---

## Why do we need both `xmolout` and `fort.7`?

ReaxFF outputs intentionally separate information:

### `xmolout`
Contains **geometry and structure**:
- atomic coordinates (x, y, z),
- atom types,
- box dimensions,
- iteration numbers.

It answers:
> *Where are the trajectories of atoms?*


### `fort.7`
Contains **chemical and electronic information**:
- partial charges,
- bond orders,
- coordination-related quantities,
- atom-resolved properties.

It answers:
> *What are the atoms doing chemically across frames?*



So, ReaxKit’s job is to **align these two views of the same simulation** in a safe,
transparent way.

---

## Re-running a multi-file CLI example

A typical command using both files looks like:

```bash
reaxkit xmolfort7 plot3d --property charge --frames 0:20:10 
```

This command:

* reads geometry from `xmolout`,
* reads per-atom charge from `fort.7`,
* aligns them frame-by-frame and atom-by-atom,
* produces a 3D scatter plot colored by charge.

---

## How ReaxKit aligns the two files
#### Frame alignment

* `xmolout` defines the master frame index.

* `fort.7` data are queried using the same frame indices.

* If a frame is missing in one file, it is skipped safely.

Internally, ReaxKit always works in frame space first, then derives
iteration or time if requested.

#### Atom alignment

* Atom indices are 0-based internally.

* `xmolout` provides atom order and coordinates.

* `fort.7` provides values keyed by atom index.

* ReaxKit matches these explicitly before analysis.

This prevents silent misalignment (a common source of errors in manual scripts).

---

## Explaining the main CLI flags
`--property charge` specifies which per-atom scalar from `fort.7` to use.

Important features:

* Alias-aware (`charge`, `q`, `partial_charge` all work).
* Validated against available columns.
* Canonicalized internally.

This allows you to focus on meaning, not file-specific naming quirks.

`--frames 0:20:10` selects which frames to analyze. This slice means:

* start at frame 0,
* stop at frame 20,
* step by 10.

Resulting frames: 0, 10, 20.

This selection applies consistently to both files.

`--atoms` (optional) allows spatial sub-selection:

* specific atoms,
* atom subsets,
* or all atoms (default).

This is especially useful for:

* focusing on active sites,
* analyzing coordination around specific species.

---
From coordination concepts to plots

Although this workflow does not explicitly compute “coordination numbers,”
it provides the building blocks for them:

geometry from xmolout,

bond-order or charge information from fort.7,

spatial projections (3D scatter, 2D heatmaps).

From here, coordination analysis typically involves:

distance cutoffs,

bond-order thresholds,

neighbor counting per atom or per species.

ReaxKit exposes these steps as analyzers so they can be reused across workflows.

---

## Plotting charges against atomic coordinates (sptial distribution of charges)

By getting:

* geometry from `xmolout`,
* bond-order or charge information from `fort.7`,

spatial projections of charges are possible (3D scatter, 2D heatmaps).

---

## Output locations

As with all workflows, outputs are organized automatically. By default:

`reaxkit_outputs/xmol_fort7/3D_scatter/`

or 

`reaxkit_outputs/xmol_fort7/2D_heatmap/`

are the output locations for the results. 

Each plot:

* is named by property and frame,
* is reproducible,
* can be regenerated with the same command.

This structure is especially helpful when exploring many frames or properties.

---

## Recommended practices for multi-file analysis

* Always let one file define the frame index (`xmolout` in here).
* Use alias names (`charge`, `q`) instead of raw column names.
* Start with sparse frame sampling, then refine.
* Export intermediate tables before heavy visualization.
* Treat geometry and chemistry as separate layers.

---

## What you can do next

With `xmolout` + `fort.7` workflows, you can now:

* compute coordination numbers explicitly,
* analyze charge transfer vs geometry,
* visualize bond-order networks,
* project properties onto 2D planes,
* combine multiple simulations consistently.

As an example, you may use:

```bash
reaxkit video make --folder reaxkit_outputs/xmol_fort7/3D_scatter
```

to make a gif video out of all frames.

This shows that multiple workflows can be sequenced to:

1. set-up simulations by generating input files such as `control`,
2. execute batch simulations,
3. perform batch analysis,
4. generate useful information comparing different simulations with different settings