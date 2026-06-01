# ReaxKit Documentation

**ReaxKit** is a Python toolkit for parsing, analyzing, and visualizing ReaxFF
simulation data.

---

## Getting started

- **[Architecture Overview](architecture_overview.md)** - Overview of different codes, how ReaxKit works, and how it stores the results
- **[Installation](installation.md)** - Install ReaxKit and dependencies
- **[Quickstart](quickstart.md)** - First end-to-end CLI run
- **[Tutorials](tutorials/index.md)** - Step-by-step workflows
- **[Examples](examples/README.md)** - Runnable scripts and sample data

---

## Core concepts

ReaxKit uses a layered model:
- **Engine I/O + generators** for data/file handling
- **Analysis tasks** for computations
- **Workflows** for CLI orchestration and presentation

API entry points:

- **[Analysis API](api/analysis/index.md)**
- **[Engine API](api/engine/index.md)**
- **[Utils API](api/utils/index.md)**
- **[Workflows API](api/workflows/index.md)**

Example current commands:
- `reaxkit timeseries --field trajectory[1].z --xaxis time`
- `reaxkit get_msd --atom-types O --xaxis time`
- `reaxkit gen_eregime --type sin --iteration-step 500 ...`
- `reaxkit gen-plot --type single --file table.csv --xaxis c1 --yaxis c2`

---

## Reference material

ReaxFF file semantics and context:
- **[ReaxFF Reference](resources/reaxff_reference/index.md)**

---

## Developer resources

- **[File templates](file_templates/index.md)**
- **[Contributing](contributing.md)**
- **[Rules and conventions](rules_and_conventions/index.md)**

---

If you are not sure where to begin, start from
**[Tutorials](tutorials/index.md)**.
