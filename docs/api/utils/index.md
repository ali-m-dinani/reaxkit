# Utils

This section documents the **utility layer** in ReaxKit.
Utilities provide **shared, low-level functionality** that is reused across handlers,
analyzers, workflows, and CLI tooling.

Unlike analysis modules, utilities do **not** encode ReaxFF-specific physics.
Instead, they handle cross-cutting concerns such as path resolution, aliasing,
unit/frame conversion, formatting, and helper logic.

---

## What belongs in `utils`

A module belongs in `reaxkit/utils/` if it:

- Is **file-format agnostic**
- Contains **no scientific interpretation**
- Is reused in multiple parts of the codebase
- Improves consistency, safety, or ergonomics

Typical responsibilities include:
- Path and output management
- Alias and name resolution
- Frame / iteration / time conversion
- Formatting and alignment helpers
- Lightweight validation and normalization

---

## Common utility modules

Below are the main utility groups typically found in this folder.

### Path and I/O helpers

- [path](path_doc.md) 
  Utilities for resolving input/output paths consistently across CLI workflows.
  Ensures files are written to the correct directory structure regardless of
  where a command is invoked.

  Typical use cases:
  - Resolving default output locations
  - Normalizing user-provided paths
  - Avoiding accidental overwrites

---

### Alias and name resolution

- [alias](alias_doc.md) 
  Maps user-friendly names and synonyms to canonical column or variable names.

  Used heavily in:
  - CLI arguments like `--xaxis`, `--yaxis`, `--col`
  - File handlers with inconsistent headers
  - Export and plotting logic

  Example:
  ```python
  resolve_alias("Density")  →  "D"
  resolve_alias("dens")     →  "D"
  ```

---

### Conversion utilities

- [convert](constants_doc.md) 
  Conversion helpers for transforming indices and units.

  Typical responsibilities:
  - Iteration → frame conversion
  - Frame → time conversion
  - Unit normalization for plotting and export

  These utilities are commonly used in workflows that accept
  `--xaxis iter|frame|time`.

---

## How utils are used in practice

Utilities are intentionally lightweight and composable.

Typical flow:

1. A **workflow** parses CLI arguments
2. A **utility** normalizes paths, aliases, or axes
3. A **handler** loads raw data
4. An **analyzer** computes results
5. Utilities assist with export or plotting

This separation keeps scientific logic clean and reusable.

---

## What utils should *not* do

Utilities should avoid:

- Parsing raw ReaxFF files
- Performing scientific calculations
- Containing CLI-specific business logic
- Importing heavy dependencies unnecessarily

Those responsibilities belong in `io/`, `analysis/`, or `workflows/`.
