# Fort83 Workflow

CLI namespace: `reaxkit fort83 <task> [flags]`

fort.83 post-processing workflow for ReaxKit.

This workflow provides a lightweight utility for processing ReaxFF `fort.83`
files, which typically record force-field optimization progress and error
information during training or fitting runs.

It supports:
- Locating the final occurrence of the marker line `Error force field` in a
  `fort.83` file.
- Extracting all subsequent lines, which usually correspond to the optimized
  force-field parameter block.
- Writing the extracted content to a new output file for reuse, inspection,
  or archiving.

The workflow is intended to streamline recovery of optimized force-field
parameters after ReaxFF training runs.

## Available tasks

### `update`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to fort.83 file |
| `--export EXPORT` | Output file name |
