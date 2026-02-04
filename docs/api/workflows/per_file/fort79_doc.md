# Fort79 Workflow

CLI namespace: `reaxkit fort79 <task> [flags]`

fort.79 sensitivity-analysis workflow for ReaxKit.

This workflow provides tools for analyzing parameter sensitivities from
ReaxFF `fort.79` files, which report how force-field parameters influence
training or objective-function errors.

It supports:
- Identifying the most sensitive (or least sensitive) force-field parameter
  based on minimum and maximum sensitivity metrics.
- Visualizing sensitivity evolution across epoch sets for a selected parameter.
- Generating tornado plots that summarize parameter influence ranges
  (min, median, and max effects) across the force field.
- Exporting processed sensitivity tables and subsets to CSV for downstream
  analysis or reporting.

The workflow is intended for diagnosing force-field training behavior,
prioritizing parameters for refinement, and interpreting sensitivity-driven
optimization results.

## Available tasks

### `most-sensitive`

#### Examples

- `reaxkit fort79 most-sensitive --plot`
- `reaxkit fort79 most-sensitive --export result.csv`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to fort.79 file. |
| `--plot` | Show plot interactively. |
| `--save SAVE` | Path or directory to save the plot image (resolved via resolve_output_path). |
| `--export EXPORT` | Export processed data to CSV (path or directory, resolved via resolve_output_path). |

### `tornado`

#### Examples

- `reaxkit fort79 tornado --top 6 --save tornado.png --export tornado.csv`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to fort.79 file. |
| `--plot` | Show plot interactively. |
| `--save SAVE` | Path or directory to save the plot image (resolved via resolve_output_path). |
| `--export EXPORT` | Export processed data to CSV (path or directory, resolved via resolve_output_path). |
| `--top TOP` | Only show/export top-N widest spans (0 = show all). |
| `--vline VLINE` | Draw a vertical reference line at x = this value. |
