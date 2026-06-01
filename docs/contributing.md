# Contributing

Thanks for your interest in contributing to ReaxKit. This guide explains how to
propose changes, set up a dev environment, run tests, and update docs.

---

## Code of Conduct

Be respectful and constructive in all project interactions.

---

## How to Contribute

### 1) Open an issue (recommended)

Before coding, open an issue describing:
- what you want to change
- why it is needed
- how you plan to implement it

### 2) Fork and clone

```bash
git clone https://github.com/ali-m-dinani/reaxkit.git
cd reaxkit
```

### 3) Create a branch

```bash
git checkout -b <branch-name>
```

### 4) Install for development

```bash
pip install -e .
```

If available:

```bash
pip install -e ".[dev]"
```

### 5) Make focused changes

- one feature/fix per PR
- avoid unrelated formatting-only edits

### 6) Run tests

```bash
pytest
```

Add/update tests for your change whenever possible.

### 7) Commit and push

```bash
git add .
git commit -m "Short summary of change"
git push origin <branch-name>
```

### 8) Open a pull request

Include:
- what changed
- why it changed
- how to test it (commands + sample inputs)

---

## What You Can Contribute

- **Handlers (I/O):** engine/file parsers (for example `src/reaxkit/engine/reaxff/io/`)
- **Analyzers:** analysis tasks and derived quantities (`src/reaxkit/analysis/`)
- **Workflows / CLI:** user-facing commands (`src/reaxkit/workflows/`)
- **Docs:** guides, references, API pages (`docs/`)
- **Examples:** runnable scripts and minimal datasets (`docs/examples/`)
- **Bug reports:** minimal reproduction + logs + expected behavior

---

## Contribution Guidelines

### Code style and structure

- [ ] **Docstrings**  
      Follow [Docstring Inclusion Rules](rules_and_conventions/docstring_content_and_inclusion_guidelines.md).

- [ ] **Code placement**  
      If analyzer logic becomes generic/reusable, move it to `utils/`.  
      If a utility becomes domain-specific, move it back to `analysis/` or a focused submodule.

- [ ] **Workflow boundaries**  
      Workflows orchestrate only.  
      Parsing belongs to handlers; computational logic belongs to analyzers.

- [ ] **Workflow command registration**  
      Register new/updated commands in CLI routing registries under `src/reaxkit/core/registry/` (analysis/generator routing as appropriate).  
      Update workflow metadata maps under `src/reaxkit/workflows/data/` when needed.

- [ ] **Shared constants / aliases / units**  
      Put shared values in central utilities/data locations and import them where needed; avoid scattering duplicate literals.

### Before submitting a PR

- [ ] The change has a clear purpose and is explained in the PR.
- [ ] Tests are added/updated where applicable.

---

## Getting Help

If you are blocked:
- open an issue with error details, OS, Python version, and minimal inputs
- include the exact command and observed output

You can also contact: `Dinani@psu.edu`.
