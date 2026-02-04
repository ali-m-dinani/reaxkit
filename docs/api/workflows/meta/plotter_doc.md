# Plotter Workflow

CLI namespace: `reaxkit plotter <task> [flags]`

General-purpose plotting workflow for ReaxKit.

This workflow provides flexible plotting utilities for arbitrary tabular data
(text, CSV, TSV, or whitespace-delimited files), without assuming any specific
ReaxFF file format or column headers.

It supports multiple plot types, including:
- single and multi-series line or scatter plots,
- directed (arrowed) line plots,
- dual y-axis plots,
- tornado plots for sensitivity-style visualization,
- 3D scatter plots with scalar coloring,
- 2D aggregated heatmaps projected from 3D data.

Columns are selected using simple 1-based column tokens (e.g. c1, c2, c3),
making the workflow suitable for rapid visualization of simulation outputs,
summaries, and post-processed analysis tables.

## Available tasks

### `directed`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to input txt/csv/tsv table. |
| `--save SAVE` | Path to save plot (file or directory). If omitted, show interactively. |
| `--title TITLE` | Optional custom plot title. |
| `--plot` | Generate and display/save the plot |
| `--xaxis XAXIS` | Single x column (e.g., 'c1'). |
| `--yaxis YAXIS` | Single y column (e.g., 'c2'). |
| `--xlabel XLABEL` | Optional x-axis label. |
| `--ylabel YLABEL` | Optional y-axis label. |

### `dual`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to input txt/csv/tsv table. |
| `--save SAVE` | Path to save plot (file or directory). If omitted, show interactively. |
| `--title TITLE` | Optional custom plot title. |
| `--plot` | Generate and display/save the plot |
| `--xaxis XAXIS` | Single x column (e.g., 'c1'). |
| `--y1 Y1` | Left y-axis column (e.g., 'c2'). |
| `--y2 Y2` | Right y-axis column (e.g., 'c3'). |
| `--xlabel XLABEL` | Optional x-axis label. |
| `--ylabel1 YLABEL1` | Optional left y-axis label. |
| `--ylabel2 YLABEL2` | Optional right y-axis label. |

### `heatmap2d`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to input txt/csv/tsv table. |
| `--save SAVE` | Path to save plot (file or directory). If omitted, show interactively. |
| `--title TITLE` | Optional custom plot title. |
| `--plot` | Generate and display/save the plot |
| `--x X` | x coordinate column (e.g., 'c1'). |
| `--y Y` | y coordinate column (e.g., 'c2'). |
| `--z Z` | z coordinate column (e.g., 'c3'). |
| `--value VALUE` | value column to aggregate (e.g., 'c4'). |
| `--plane {xy,xz,yz}` | Projection plane for heatmap (default: xy). |
| `--bins BINS` | Grid resolution: int (e.g., 50) or 'nx,ny' (e.g., '50,100'). |

### `scatter3d`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to input txt/csv/tsv table. |
| `--save SAVE` | Path to save plot (file or directory). If omitted, show interactively. |
| `--title TITLE` | Optional custom plot title. |
| `--plot` | Generate and display/save the plot |
| `--x X` | x coordinate column (e.g., 'c1'). |
| `--y Y` | y coordinate column (e.g., 'c2'). |
| `--z Z` | z coordinate column (e.g., 'c3'). |
| `--value VALUE` | Value column for coloring (e.g., 'c4'). |

### `single`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to input txt/csv/tsv table. |
| `--save SAVE` | Path to save plot (file or directory). If omitted, show interactively. |
| `--title TITLE` | Optional custom plot title. |
| `--plot` | Generate and display/save the plot |
| `--xaxis XAXIS` | Comma-separated list of x columns (e.g., 'c1' or 'c1,c3'). |
| `--yaxis YAXIS` | Comma-separated list of y columns (e.g., 'c2' or 'c2,c4'). |
| `--xlabel XLABEL` | Optional x-axis label. |
| `--ylabel YLABEL` | Optional y-axis label. |
| `--scatter` | Use scatter instead of line plot. |

### `tornado`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to input txt/csv/tsv table. |
| `--save SAVE` | Path to save plot (file or directory). If omitted, show interactively. |
| `--title TITLE` | Optional custom plot title. |
| `--plot` | Generate and display/save the plot |
| `--label LABEL` | Column for labels (e.g., 'c1'). |
| `--min MIN` | Column for minimum values (e.g., 'c2'). |
| `--max MAX` | Column for maximum values (e.g., 'c3'). |
| `--median MEDIAN` | Optional column for median values (e.g., 'c4'). |
| `--top TOP` | Show only top-N widest bars (0 = all). |
| `--vline VLINE` | Optional vertical reference line (e.g., 0.0). |
| `--xlabel XLABEL` | Optional x-axis label. |
| `--ylabel YLABEL` | Optional y-axis label. |
