# Meta Plotting with `gen-plot`

This tutorial updates the older `plotter ...` style to the current
presentation workflow command:

```bash
reaxkit gen-plot --type <plot_type> ...
```

`gen-plot` is file-agnostic: it reads tabular files and plots by column
selectors (`c1`, `c2`, ...), without requiring ReaxFF-specific semantics.

---

## Supported plot types

- `single`
- `directed`
- `dual`
- `tornado`
- `scatter3d`
- `heatmap2d`

All are selected through `--type`.

---

## Input model

`gen-plot` accepts text/CSV/TSV tables:
- CSV/TSV or whitespace-delimited text
- optional comment lines with `#`
- columns referenced with 1-based tokens (`c1`, `c2`, ...)

---

## Core examples

### 1) Single line/scatter plot

```bash
reaxkit gen-plot --type single --file summary.txt --xaxis c1 --yaxis c2 --plot
```

Multi-series with shared x:

```bash
reaxkit gen-plot --type single --file summary.txt --xaxis c1 --yaxis c2,c3,c4 --plot
```

Pairwise x-y mode:

```bash
reaxkit gen-plot --type single --file data.txt --xaxis c1,c3 --yaxis c2,c4 --plot
```

Use markers instead of lines:

```bash
reaxkit gen-plot --type single --file data.txt --xaxis c1 --yaxis c2 --scatter --plot
```

### 2) Directed curve

```bash
reaxkit gen-plot --type directed --file summary.txt --xaxis c1 --yaxis c2 --save directed.png
```

### 3) Dual y-axis

```bash
reaxkit gen-plot --type dual --file summary.txt --xaxis c1 --y1 c2 --y2 c3 --save dual_plot.png
```

### 4) Tornado

```bash
reaxkit gen-plot --type tornado --file results.txt --label c1 --min c2 --max c3 --median c4 --top 10 --save tornado.png
```

### 5) 3D scatter

```bash
reaxkit gen-plot --type scatter3d --file points.txt --x c1 --y c2 --z c3 --value c4 --save scatter3d.png
```

### 6) 2D heatmap from 3D coordinates

```bash
reaxkit gen-plot --type heatmap2d --file points.txt --x c1 --y c2 --z c3 --value c4 --plane xz --bins 100,80 --save heatmap_xz.png
```

---

## Output behavior

- `--plot` displays interactively
- `--save` writes figure to file
- you can use both
- if neither is given, no plot artifact is produced

---

## Data cleaning behavior

Before plotting, selected columns are coerced to numeric (where applicable)
and invalid rows are dropped. If no valid rows remain, the command exits with
a clear error.

---

## Handoff to video generation

When you save frame sequences, use:

```bash
reaxkit gen-video --folder <frames_dir> --output output.mp4 --fps 10
```

This keeps plotting (`gen-plot`) and video composition (`gen-video`) as two
clean, composable steps.

---

[Back to Tutorials](index.md)
