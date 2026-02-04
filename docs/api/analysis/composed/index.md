# Composed Analyses

This section documents **composed analyses** in ReaxKit — higher‑level routines that
**combine data from multiple ReaxFF files** to compute physically meaningful,
structure‑aware, or system‑level quantities.

These analyses sit *above* per‑file analyses and represent the main scientific value
layer of ReaxKit.

---

## What “composed” means in ReaxKit

A *composed analysis*:

- Uses **two or more ReaxFF files**
- Combines data from **multiple handlers**
- Encodes domain logic (physics / chemistry / materials insight)
- Often operates on **local environments, clusters, or time‑evolving structures**
- Is exposed via CLI workflows that coordinate several handlers internally

Typical examples:
- Coordinates (`xmolout`) + connectivity (`fort.7`)
- Energies (`energylog`) + volume (`fort.74`)
- Trajectory (`xmolout`) + charges + connectivity → dipoles / polarization

---

## Common composed analysis patterns

### Structure + connectivity

- Local coordination environments
- Molecule / cluster identification
- Bond‑aware geometric properties
- Per‑atom or per‑cluster observables

**Typical inputs**
- `xmolout`
- `fort.7`

---

### Trajectory + derived physics

- Dipole moments and polarization
- Mean‑square displacement by species or cluster
- Time‑resolved local observables

**Typical inputs**
- `xmolout`
- `fort.7`
- (optionally) charge or electrostatics data

---

### Thermodynamics + mechanics

- Energy–volume relations
- Bulk modulus fits
- Stress–strain analysis

**Typical inputs**
- `energylog` / `fort.73`
- `fort.74`
- `fort.76`
- `fort.99`

---

## What each composed analysis page contains

Each page typically documents:

1. **Scientific objective**
2. **Required input files**
3. **Handlers and analyzers used**
4. **Data‑flow diagram (conceptual)**
5. **Python API example**
6. **CLI workflow usage**
7. **Exported outputs**
8. **Physical assumptions and caveats**

---

## When to start from per‑file analyses instead

If you only need:

- Raw trajectories
- Energies vs iteration
- Single‑file summaries or plots

→ Start in [per-file analysis](../per_file/index.md) instead.

Composed analyses intentionally assume familiarity with ReaxFF outputs and build on
clean, validated per‑file data.
