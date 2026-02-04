# Introspection Workflow

CLI namespace: `reaxkit introspection <task> [flags]`

CLI workflow for introspecting ReaxKit modules and folders.

This workflow powers the `reaxkit intspec` command, allowing users to:
- Inspect a single Python module and view its top-level docstring summary
  along with public functions/classes and their one-line descriptions.
- Recursively scan a folder or package and list all contained `.py` files
  with their module docstring first lines.

It is designed as a lightweight discovery and navigation tool to help users
understand what functionality exists inside ReaxKit without opening files
manually.

## Available tasks

### `run`

Introspect a module (--file) or folder (--folder).

#### Examples

- `reaxkit intspec --folder workflow`
- `reaxkit intspec run --folder workflow`
- `reaxkit intspec --file fort7_analyzer`
- `reaxkit intspec run --file fort7_analyzer`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Module name (e.g. fort7_analyzer) or path to .py |
| `--folder FOLDER` | Folder/package (e.g. workflow, workflows, reaxkit/workflows) |
