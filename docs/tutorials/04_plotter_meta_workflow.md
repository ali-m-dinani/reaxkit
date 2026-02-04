# Meta Plotting with `plotter`: Visualizing Any Tabular Data

So far, the tutorials focused on **ReaxFF-specific workflows** that understand
file semantics (`xmolout`, `fort.7`, etc.).  
In this tutorial, we introduce a different kind of workflow: a **meta workflow**.

The `plotter` workflow is intentionally **file-agnostic**. It operates on *any*
tabular data file and lets you generate common scientific plots directly from
the CLI—without writing Python.

This makes it ideal for:
- quick inspection of exported CSV files,
- plotting intermediate results,
- visualizing external or post-processed data,
- avoiding one-off matplotlib scripts.

---

## What makes `plotter` a *meta* workflow?

Unlike other workflows:

- it does **not** use ReaxFF handlers,
- it does **not** assume file-specific semantics,
- it works purely on **columns**.

In other words:
> `plotter` does not care *where* the data came from — only how it is arranged.

This is why it lives under **meta workflows**, alongside tools like help,
introspection, and generic utilities.

---

## Input data model

The `plotter` workflow expects a **plain text table**:

- whitespace-, CSV-, or TSV-delimited,
- optional comment lines starting with `#`,
- numeric data in columns.

Important design choice:
- **No header assumptions** are made.
- Columns are addressed by position, not name.

Example file (conceptual):

```
iter strain stress

0 0.00 0.0
10 0.01 1.2
20 0.02 2.4
```


These columns would be referenced as:
- `c1` → first column
- `c2` → second column
- `c3` → third column

---

## Column selectors: `c1`, `c2`, …

All plotter commands use **1-based column selectors**:

- `c1` → column 1
- `c3` → column 3

Internally, ReaxKit converts these to 0-based indices and validates them
against the input file.

This design:
- avoids header parsing issues,
- works reliably on mixed-format files,
- keeps CLI usage explicit.

---

## Single-curve and multi-curve plots

#### Basic single plot

```bash
 reaxkit plotter single --file summary.txt --xaxis c1 --yaxis c2 --plot
```

This produces a standard line plot of `c2` vs `c1`.

#### Multiple y columns vs one x

```bash
reaxkit plotter single --file summary.txt --xaxis c1 --yaxis c2,c3,c4 --plot
```

Here:

* `c1` is used as the common x-axis,
* each y column is plotted as a separate series.


#### Pairwise x–y plotting

```bash
reaxkit plotter single --file data.txt --xaxis c1,c3 --yaxis c2,c4 --plot
```

This produces:

* (`c2` vs `c1`)
* (`c4` vs `c3`)

Pairing is enforced to prevent accidental mismatches.


#### Directed plots

Directed plots show directionality along a curve, useful for trajectories
or hysteresis-like data.

```bash
reaxkit plotter directed --file summary.txt --xaxis c1 --yaxis c2 --save directed.png
```

This highlights progression along the path, not just shape.


#### Dual y-axis plots

To compare two quantities with different scales:

```bash
reaxkit plotter dual --file summary.txt --xaxis c1 --y1 c2 --y2 c3 --save dual_plot.png
```

This produces:

* left y-axis → `c2`
* right y-axis → `c3`

#### Tornado plots (sensitivity-style visualization)

Tornado plots visualize ranges (min/max, optional median):

```bash
reaxkit plotter tornado --file results.txt --label c1 --min c2 --max c3 --median c4 --top 10 --save tornado.png
```

Typical use cases:

* sensitivity analysis,
* uncertainty ranges,
* parameter importance ranking.


#### 3D scatter plots

For spatial or parametric data:

```bash
reaxkit plotter scatter3d --file points.txt --x c1 --y c2 --z c3 --value c4 --save scatter3d.png
```

This plots (x, y, z) points colored by value.


#### 2D heatmaps from 3D data

To project 3D data onto a plane:

```bash
reaxkit plotter heatmap2d --file points.txt --x c1 --y c2 --z c3 --value c4 --plane xz --bins 100,80 --save heatmap_xz.png
```
 
Here:

* data are projected onto the `xz` plane,
* values are aggregated into bins,
* resolution is user-controlled.

---

## Data cleaning and validation (important)

Internally, plotter:

* coerces selected columns to numeric,
* drops rows with invalid values,
* refuses to plot if no valid data remain.

This prevents misleading plots from malformed input.

---

## Output behavior

As with other workflows:

* `--plot` → display interactively
* `--save` → write to file (no display required)

If neither is specified, the workflow exits early with a clear message.

---

## What you can do next

With the plotter meta workflow, you can now:

* visualize outputs from any ReaxKit workflow,
* inspect external datasets quickly,
* prototype plots before writing scripts,
* integrate ReaxKit into broader analysis pipelines.





