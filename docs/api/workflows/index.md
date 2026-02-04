# Workflows

This section documents **ReaxKit workflows**.
Workflows are the **user-facing layer** of the toolkit, exposing analysis functionality
through a consistent and discoverable CLI interface.

They connect handlers and analyzers into complete, runnable tasks.

---

## What is a workflow in ReaxKit

A workflow:

- Parses CLI arguments
- Loads data via one or more handlers
- Executes analysis logic
- Optionally plots or exports results

Most workflows follow the pattern:

```bash
reaxkit <workflow> <task> [options]
```

---

## Workflow categories

ReaxKit workflows are organized into three groups:

### Per-file workflows
Operate on **a single ReaxFF file** and provide direct access to file-specific analyses.

→ See [per-file workflows](per_file/index.md)

---

### Composed workflows
Coordinate **multiple files and analyses** to perform higher-level scientific tasks.

→ See [composed workflows](composed/index.md)

---

### Meta workflows
Provide **tooling and support utilities** such as help, introspection, plotting,
and media generation.

→ See [meta workflows](meta/index.md)

---

## When to start here

If you are new to ReaxKit, workflows are the best entry point:
they expose most functionality without requiring Python scripting.

For lower-level APIs, see the [io](../io/index.md) and [analysis](../analysis/index.md) documentation instead.
