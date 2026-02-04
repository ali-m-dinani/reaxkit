# ReaxKit Templates for developers

This directory contains **minimal, reference templates** for developing new
handlers, analyzers, and workflows in ReaxKit.  
They are **not** runtime code and are intended for **contributors and developers**
who want to extend ReaxKit in a consistent, maintainable way.

The templates illustrate ReaxKit’s core architectural principles:
**separation of concerns**, **tidy data flow**, and **CLI-first design**.

---

## Overview of Templates

### 1. `template_handler.py` — File Parsers (I/O layer)

Use [template_handler.py](https://github.com/ali-m-dinani/reaxkit/blob/master/docs/file_templates/template_handler.py)
when implementing a **new ReaxFF file handler**.

A handler’s responsibility is strictly limited to:
- reading a raw file,
- parsing it into a **summary DataFrame**,
- optionally extracting **per-frame data**,
- performing lightweight cleaning (e.g. duplicate iterations),
- exposing a consistent API (`dataframe()`, `frame()`, `iter_frames()`).

**Handlers must NOT:**
- give data to users through `get()` functions
- perform numerical analysis,
- generate plots,
- make scientific interpretations.

This guarantees that handlers remain reusable across analyzers, workflows,
CLI commands, and future GUIs.

---

### 2. `template_analyzer.py` — Data Analysis (analysis layer)

Use [template_analyzer.py](https://github.com/ali-m-dinani/reaxkit/blob/master/docs/file_templates/template_analyzer.py) to implement **analysis routines** that operate on data
provided by handlers.

Analyzers:
- accept one or more handlers as input,
- retrieve data via `handler.dataframe()` or `handler.frame(...)`,
- compute derived quantities,
- return results as **tidy pandas objects**.

Naming guidelines:
- use descriptive, intention-revealing names (`compute_x`, `extract_y`,
  `calculate_metric`)
- avoid embedding file-specific logic (that belongs in handlers).

To help `reaxkit intspec` CLI command find and report what functions are available in each analyzer, please put a one line short description of that function in the first line of its docstring, immediately followed by tripple quote signs.

Analyzers should remain **pure and testable**, with no CLI or plotting logic.

---

### 3. `template_workflow.py` — CLI Tasks (workflow layer)

Use [template_workflow.py](https://github.com/ali-m-dinani/reaxkit/blob/master/docs/file_templates/template_workflow.py) to expose analysis functionality through the **ReaxKit CLI**.

Workflows:
- define CLI subcommands and arguments,
- instantiate handlers,
- call analyzers,
- optionally trigger plotting or export,
- handle user-facing messages and output paths.

Key conventions shown in the template:
- `reaxkit <kind> <task> --flags` CLI structure
- grouping common CLI arguments via helper functions
- consistent `--plot`, `--save`, and `--export` semantics
- examples embedded in CLI help text
- all outputs routed to a structured `reaxkit_output/` directory

Workflows should **orchestrate**, not compute.

---

## Typical Development Flow

When adding support for a new file or analysis:

1. **Write a handler** (based on `template_handler.py`)
2. **Write one or more analyzers** (based on `template_analyzer.py`)
3. **Expose them via a workflow** (based on `template_workflow.py`)
4. Add tests and documentation as needed

This layered approach keeps ReaxKit modular, debuggable, and scalable.

---

## Notes

- These templates are intentionally minimal.
- Copy them when starting new development; do not import them directly.
- They reflect current best practices used throughout the ReaxKit codebase.

If in doubt, search existing handlers, analyzers, or workflows.
