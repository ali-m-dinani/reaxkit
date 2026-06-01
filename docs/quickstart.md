# Quick Start

This page walks through a minimal end-to-end ReaxKit CLI run.

---

## Step 1: Verify installation

```bash
reaxkit --help
```

If help is shown, the CLI is available.

---

## Step 2: Discover commands and searchable topics

```bash
reaxkit help
reaxkit help pressure
reaxkit help "electric field"
```

Optional introspection examples:

```bash
reaxkit intspec --folder workflow
reaxkit intspec --file trajectory_workflow
```

---

## Step 3: Check task-specific help

Current ReaxKit uses direct command workflows. For example:

```bash
reaxkit timeseries -h
reaxkit get_msd -h
```

Use `-h` on any command to inspect supported flags and examples.

---

## Step 4: Run a simple analysis

Extract atom-1 z trajectory and export CSV:

```bash
reaxkit timeseries --field trajectory[1].z --xaxis time --export atom1_z.csv
```

What this does:
- resolves trajectory data input
- computes the requested time series
- exports a table to CSV

---

## Output behavior

Depending on flags and command:
- tables can be printed to terminal
- plots can be displayed (`--show`) or saved (`--save`)
- tables can be exported (`--export`)
- generated input files are written under generator output locations

---

## What happened internally

At a high level:
1. workflow parses CLI args
2. runtime resolves required data sources
3. analysis task executes
4. presentation/export layer handles outputs

---

## Next steps

- Detailed walkthrough: [01_understanding_quickstart.md](tutorials/01_understanding_quickstart.md)
- More runnable examples: [examples](examples/README.md)
- Developer templates: [file templates](file_templates/index.md)
- Full tutorial sequence: [tutorials](tutorials/index.md)
