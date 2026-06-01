# ReaxFF input files

This directory documents the **input files used by ReaxFF**.  
ReaxFF input files can be divided into **mandatory files**, which are required for any run, and **optional files**, which are only needed for specific simulation modes or advanced workflows such as force field optimization.

This reference focuses on the **file formats and their roles**, independent of ReaxKit-specific tooling.

---

## Overview of ReaxFF input files

The table below summarizes all commonly used ReaxFF input files, their role, and whether they are required.

### Table 1 — ReaxFF input files

| File name | Category | Short description |
|---|---|---|
| `geo` | Mandatory | System geometry |
| `ffield` | Mandatory | Force field parameters |
| `control` | Mandatory | Run-control switches |
| `exe` | Mandatory | Execution script used to launch ReaxFF |
| `models.in` | Optional | Geometry file locations |
| `tregime.in` | Optional | Temperature regime definition |
| `vregime.in` | Optional | Volume / cell-parameter regime definition |
| `eregime.in` | Optional | Electric field definition |
| `iopt` | Optional | Toggle between normal run and force field optimization |
| `addmol.bgf` | Optional | Molecular fragment definition for on-the-fly insertion |
| `charges` | Optional | Fixed atomic charges |
| `vels` | Optional | MD restart file |
| `trainset.in` | Force field optimization | Training set / cost-function definition |
| `params` | Force field optimization | Optimizable force field parameters |
| `koppel2` | Force field optimization | Linked force field parameters |

---

## File categories

### Mandatory files
These files must be present for **any ReaxFF run**:
- `geo`
- `ffield`
- `control`
- `exe`

Without these files, ReaxFF will not start.

---

### Optional simulation-control files
These files enable advanced simulation protocols:
- `tregime.in` — temperature control
- `vregime.in` — strain and volume control
- `eregime.in` — external electric fields
- `addmol.bgf` — molecule insertion (GCMD-style simulations)
- `charges` — fixed-charge simulations
- `vels` — MD restarts
- `models.in` — geometry management

---

### Force field optimization files
These files are only required when **optimizing ReaxFF force fields**:
- `iopt`
- `trainset.in`
- `params`
- `koppel2`

Together, they define:
- what data is fitted,
- which parameters are optimized,
- how parameters are linked,
- and how the optimization is executed.

---

## How to use this reference

Each file listed above has a dedicated Markdown page in this directory describing:
- its purpose,
- file format,
- examples,
- and common usage patterns.

Start with the mandatory files, then explore optional and optimization-related files as needed.
