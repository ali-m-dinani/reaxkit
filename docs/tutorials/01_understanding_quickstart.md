# Understanding the Quick Start

In [quickstart.md](../quickstart.md), you ran a working ReaxKit CLI command and obtained results
with almost no setup. This tutorial explains **what actually happened** during
that process and introduces several core concepts that will help you use
ReaxKit effectively.

We will walk through the *same command* step by step and explain:
- how ReaxKit processes an `xmolout` file,
- what each CLI flag means,
- how handlers, analyzers, and workflows interact,
- the difference between **frame**, **iteration**, and **time**,
- where outputs are written,
- and what you can do next.

---

## Re-running the Quick Start command

From the Quick Start, the example command was:

```bash
reaxkit xmolout trajget --atoms 1 --dims z --xaxis time --export atom1_z.csv
```

This command extracts the z-coordinate trajectory of atom 1 from an
xmolout file and exports it as a CSV table.

Let’s unpack what each part does.

---

## Explaining the CLI flags

`--atoms 1` selects which atoms to extract trajectories for. 

* Atom indices are 1-based (as in ReaxFF input/output), hence `--atoms 1` means only atom 1.

* You can also use:

    * `--atoms 1,5,12`

    * `--atoms 1:10`

Internally, ReaxKit converts these to 0-based indices (to properly work with pandas dataframes) before accessing the underlying arrays.

`--dims z` specifies which coordinate components to include.

* Allowed values are: x, y, z

* `--dims z` means: only extract the z-coordinate.

If omitted, all three coordinates (x y z) are included.

`--xaxis time` controls what appears on the x-axis (and the first column of exported data).

Options:

* `frame` → frame index (0, 1, 2, …)

* `iter` → MD iteration number (from the file)

* `time` → physical time (which will be calculated using time-step in the `control` file)

Internally:

* ReaxKit always starts from frame indices

* `--xaxis` simply converts them to the requested representation

This conversion happens in a single, centralized utility (i.e., `src/reaxkit/utils/convert.py`), so all workflows
behave consistently.

`--export atom1_z.csv` exports the processed data to a CSV file.

* Exported files are automatically placed in a structured output directory
(see below).

---

## What happens internally: the pipeline

When you run the command above, ReaxKit executes a multi-layer pipeline:

1. Workflow gets the CLI command, and extracts the flags, settings, requests
2. An specific analyzer is called to process the request
3. A handler is called to parse the appropriate file(s)
4. A specific calculation is done on the parsed data by the analyzer
5. Results are sent back to the workflow, and workflow shows the results to the user

Below, the above parts are explained more.

### 1. Workflow (CLI layer)

The workflow:

* `cli.py` parses CLI arguments and decides what task to run (`trajget`),

* `xmolout_workflow` orchestrates I/O, plotting, and export.

Workflows do not parse files or perform scientific calculations.

### 2. Handler (I/O layer)

The `XmoloutHandler`:

* reads the raw `xmolout` file,

* parses per-frame metadata (iterations, energies, cell dimensions),

* stores per-frame atom tables (coordinates, atom types),

* exposes a consistent interface:

    * `dataframe()` → per-frame summary

    * `frame(i)` → atom coordinates and types for frame i

Handlers are intentionally lightweight and reusable.

### 3. Analyzer (analysis layer)

The analyzer functions:

* request data from the handler,

* apply selections (frames, atoms, dimensions),

* construct tidy pandas tables (long or wide format),

* return results without performing any I/O.

In this example, the analyzer builds a table with:

* frame_index

* iter

* atom_id

* selected coordinates (z)

---

## Frame vs iteration vs time (important)

This distinction is critical when analyzing ReaxFF trajectories.

#### Frame

* A frame is simply the n-th snapshot in the trajectory.

* Frames are always 0-based and sequential.

* Frames are the internal reference used by ReaxKit.

#### Iteration (`iter`)

* The MD iteration number stored in the `xmolout` file.

* Iterations may:

    * skip values,

    * repeat,

    * restart (e.g., after restarts).

ReaxKit always preserves `iter` explicitly.

#### Time

* Physical time derived from `iteration` × `timestep`.

* Only available if the `control` file is available.

#### Best practice:
* Use frame for indexing and selection, and use iter or time for plotting
and interpretation

---

## Output locations

ReaxKit never writes files randomly. By default:

* exported CSV files go to: `reaxkit_outputs/<workflow_name>/`


* generated inputs or derived files go to: `reaxkit_generated_inputs/`


For example: 
* `reaxkit_outputs/xmolout/atom1_z.csv` is for a trajectory result based on a `xmolout` file.
* `reaxkit_generated_inputs/vregime.in` is for a generated template for a `vregime.in` file.


This structure:

* prevents clutter,

* keeps results grouped by workflow,

* makes batch analysis reproducible.

You can override paths explicitly if needed by passing the desired **directory** (not the file name) to the `save`, `export`, or `output` flag.

---

## Recommended practices

* Start with currently developed CLI workflows, not Python scripts.
* Prefer frame for selection, iter/time for plotting.
* Keep handlers simple; do analysis in analyzers.
* Treat workflows as orchestration, not computation.

---

## What you can do next

Now that you understand the Quick Start pipeline, you can:

* extract trajectories for multiple atoms or atom types,
* compute mean squared displacement (MSD), radial distribution functions (RDF), etc.,
* combine `xmolout` with other files (e.g. fort.7) for more advanced analysis (i.e., dipole moment calculations),
* explore advanced plotting and export options.

Next tutorials will build on this foundation and introduce
multi-file analysis and more advanced workflows.
















