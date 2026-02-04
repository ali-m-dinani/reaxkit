# ReaxFF File Reference Index

This section of the documentation provides a **structured, tool-agnostic reference**
to **ReaxFF input and output files**.

It is intended to help users understand how ReaxFF simulations are **configured,
executed, and analyzed**, independently of any specific tooling (including ReaxKit),
although ReaxKit parsers and analyzers often build directly on these files.

> **Provenance note**  
> This Markdown documentation was created on **February 2, 2026**, based on the
> *Reax_user_manual_June_2017_full_version.pdf*.

---

## How this reference is organized

The documentation is split into two main parts:

- **Input files** — define what is simulated and how it is run  
- **Output files** — contain trajectories, energies, diagnostics, and optimization data

Each file has its **own dedicated page** describing:
- purpose
- format
- when it is produced or required
- common usage notes

This reference focuses on **file purpose and format**, not on ReaxKit-specific workflows.

---

## Input files

ReaxFF input files define **what is simulated and how it is run**. A full list of them are at [`input_files/README.md`](input_files/README.md).

### Core (mandatory)
- [`geo`](input_files/geo.md) — system geometry
- [`ffield`](input_files/ffield.md) — force-field parameters
- [`control`](input_files/control.md) — run-control settings
- [`exe`](input_files/exe.md) — execution script

### Optional simulation control
- [`tregime.in`](input_files/tregime.md) — temperature regimes
- [`vregime.in`](input_files/vregime.md) — cell / volume control
- [`eregime.in`](input_files/eregime.md) — electric-field control
- [`addmol.bgf`](input_files/addmol.md) — molecule insertion
- [`charges`](input_files/charges.md) — fixed charges
- [`vels`](input_files/vels.md) — MD restart

### Force-field optimization
- [`iopt`](input_files/iopt.md) — optimization toggle
- [`trainset.in`](input_files/trainset.md) — training-set definition
- [`params`](input_files/params.md) — optimizable parameters
- [`koppel2`](input_files/koppel2.md) — parameter linking

---

## Output files

ReaxFF output files contain **trajectories, energies, diagnostics,
and force-field optimization data**.

- [`xmolout`](output_files/xmolout.md) — atomic trajectories
- [`fort.7 / fort.8`](output_files/fort7_fort8.md) — connectivity and charges
- [`fort.57`](output_files/fort57.md) — MM minimization report
- [`fort.58`](output_files/fort58.md) — partial MM energies
- [`fort.71`](output_files/fort71.md) — MD energy, temperature, pressure
- [`fort.73`](output_files/fort73.md) — MD energy contributions
- [`fort.13`](output_files/fort13.md) — total force-field error
- [`fort.79`](output_files/fort79.md) — parabolic extrapolation
- [`fort.99`](output_files/fort99.md) — detailed cost-function report

Additional output files are summarized in the
[`output_files/README.md`](output_files/README.md).

---

## Included reference manuals

In addition to the Markdown documentation, this directory contains the original
ReaxFF user manuals:

- **Reax_user_manual_June_2017_full_version.pdf**  
  The complete historical ReaxFF manual.

- **Reax_user_manual_June_2017_trimmed_version.pdf**  
  A trimmed version with input/output file descriptions removed.

### Recommended reading path

For **new users**:

1. Start with this Markdown reference to understand **ReaxFF input and output files**.
2. Use the *trimmed* PDF for:
   - ReaxFF concepts and theory
   - potential functions
   - program structure
   - performance considerations
   - supported execution environments
   - the e-ReaxFF method
   - background literature

Use the *full* PDF only if historical or low-level detail is required.

---

## When to use this reference

Use this section if you want to:

- Understand **raw ReaxFF file formats**
- Interpret ReaxFF output without post-processing tools
- Debug simulations or force-field optimization runs
- Learn how different ReaxFF files interact
- Cross-check ReaxKit results against native ReaxFF output

**Where to start:**
- New simulations → **mandatory input files**
- Force-field training → **optimization files**
- Analysis workflows → **output files**

---

## Notes

- File availability depends on **run type** (MM, MD, FF optimization).
- Output frequency is typically controlled by keywords in `control`.
- Many output files are overwritten unless append behavior is enabled.

This page serves as the **single entry point** for all ReaxFF file documentation.
