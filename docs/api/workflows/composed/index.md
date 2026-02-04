# Composed Workflows

This section documents **composed workflows** in ReaxKit.
Composed workflows orchestrate **multiple handlers and analyses** to perform
higher-level tasks that reflect real scientific use cases, based on [composed analyzers](../../analysis/composed/index.md).

They are typically the entry point when a task cannot be completed from a
single ReaxFF file alone.

---

## What makes a workflow “composed”

A composed workflow:

- Uses **multiple ReaxFF files**
- Coordinates **several handlers and analyzers**
- Encodes a complete analysis pipeline
- Exposes a single, user-facing CLI command

Typical examples include:
- Coordinate + connectivity–based analyses
- Energy–volume or stress–strain pipelines
- Local or cluster-resolved property calculations

---

## How composed workflows are used

From the CLI, composed workflows look just like per-file ones:

```bash
reaxkit <workflow> <task> [options]
```

Internally, they:
1. Load multiple files via handlers
2. Run one or more analyses
3. Optionally plot or export results

---

## What each workflow page documents

Each composed workflow page usually includes:

- Scientific objective
- Required input files
- High-level data flow
- Available tasks and CLI usage
- Output formats (plots, CSV, etc.)

---

## When to use composed workflows

Use composed workflows when your analysis depends on:
- Structural context
- Connectivity or clustering
- Multiple physical observables combined

If your task only needs one file, see **per-file workflows** instead.

---

Composed workflows represent the **end-to-end analysis layer** of ReaxKit,
bridging raw ReaxFF outputs to physically meaningful results.
