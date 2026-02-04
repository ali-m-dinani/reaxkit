# Contributing

Thanks for your interest in contributing to ReaxKit! This guide explains how to propose changes, set up a dev environment, run tests, and update documentation.

---

## Code of Conduct

Please be respectful and constructive in all interactions related to this project.

---

## How to Contribute

### 1. Open an Issue (recommended)
Before starting work, open an issue to describe:
- what you want to change
- why it’s needed
- how you plan to implement it

This helps avoid duplicated effort and aligns expectations.

### 2. Fork and Clone
Fork the repository by clicking on the 'Fork' button in the top right corner of the repository page. This will create a copy of this repository in your GitHub account.

Then, clone the forked repository to your local machine by opening a terminal and running the following command

```bash
git clone https://github.com/ali-m-dinani/reaxkit.git
cd reaxkit
```

### 3. Create a Branch
Create a feature branch:
```bash
git checkout -b <branch-name>
```

### 4. Install for Development
ReaxKit uses a `src/` layout. Install in editable mode:

```bash
pip install -e .
```

If your project defines dev extras (recommended), use:
```bash
pip install -e ".[dev]"
```

### 5. Make Changes
Keep changes focused and small when possible:
- one feature/fix per PR
- avoid unrelated formatting-only changes

### 6. Run Tests
If you have a test suite (recommended: `pytest`), run:
```bash
pytest
```

If no tests exist for your change, please add them when possible.

### 7. Commit and Push
Write clear commit messages:
```bash
git add .
git commit -m "Short summary of change"
git push origin <branch-name>
```

### 8. Open a Pull Request
Open a PR against the main branch and include:
- what changed
- why it changed
- how to test it (commands + example inputs)

---

## What You Can Contribute

Common contribution areas:

- **Handlers (I/O)**: parsers for ReaxFF input/output files (`src/reaxkit/io/`)
- **Analyzers**: analysis routines and derived quantities (`src/reaxkit/analysis/`)
- **Workflows / CLI**: user-facing commands (`src/reaxkit/workflows/`)
- **Docs**: guides, references, and API docs (`docs/`)
- **Examples**: runnable scripts and minimal datasets (`docs/examples/`)
- **Bug reports**: minimal reproducible examples, error logs, and expected behavior

---

## Contribution Guidelines

To ensure high-quality and maintainable contributions, please follow the guidelines below.

### Code Style and Structure

- [ ] Docstrings  
  Please follow the conventions described in  [Docstring Guidelines](file_templates/docstrings.md).

- [ ] Code placement  
  - If you implement a function inside an analyzer and later realize it is **not file-specific** and can be reused elsewhere, move it to `utils/` instead of keeping it under `analysis/`.  
  - If a utility becomes domain- or analysis-specific over time, consider moving it back into `analysis/` or a dedicated submodule.

- [ ] **Workflows**  
  When adding or modifying a workflow, ensure that:
  - [ ] Workflows are used **only for orchestration**. Parsing belongs in `handler` modules, and computational logic belongs in `analyzer` modules.
  - [ ] The workflow module is imported in `cli.py`, which serves as the central dispatcher.
  - [ ] The workflow’s Python file name is registered in `cli.py` under `WORKFLOW_MODULES` as a key–value pair.

- [ ] Constants, alias, or units
  - Instead of just adding constants to a developing file, add them to the corresponding files in 
  the `utils/` folder, and then use `alias.py`, `constants,py`, and `units.py` under the 
  `utils` directory.
  

### Before Submitting a Pull Request

Please verify that:

- [ ] The change has a **clear purpose** and is well explained in the PR description.
- [ ] Tests are **added or updated** where applicable.

Following these guidelines helps keep ReaxKit consistent, modular, and easy to extend.


---

## Getting Help

If you get stuck:

- open an issue with details (error message, OS, Python version, minimal input files)
- include the command you ran and the expected output

You can also send your problem directly to `Dinani@psu.edu`.