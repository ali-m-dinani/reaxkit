# Meta Workflows

This section documents **meta workflows** in ReaxKit.
Meta workflows do not analyze ReaxFF simulation data directly.
Instead, they provide **tooling, introspection, and orchestration utilities**
that help users explore, understand, and present ReaxKit’s capabilities.

They operate *around* analyses rather than *on* simulation data.

---

## What makes a workflow “meta”

A meta workflow typically:

- Does **not** require ReaxFF output files
- Operates on ReaxKit’s internal structure, metadata, or user inputs
- Improves usability, discoverability, or presentation
- Exposes helper commands rather than scientific analyses

---

## Available meta workflows

Common meta workflows include:

- **help**  
  Interactive and searchable help system for workflows, files, and variables.

- **introspection**  
  Programmatic inspection of available handlers, analyzers, workflows, and tasks.

- **plotter**  
  Generic plotting utilities for tabular data, independent of file origin.

- **make_video**  
  Assemble image frames or plots into videos or animations.

---

## When to use meta workflows

Use meta workflows when you want to:

- Discover what ReaxKit can do
- Inspect available commands or analyses
- Plot or visualize pre-existing data
- Generate media outputs without running a full analysis

---

Meta workflows form the **support and tooling layer** of ReaxKit,
making the rest of the toolkit easier to explore and use.
