# `fort.73` file — MD partial energy decomposition

The **`fort.73`** file contains a **partial energy contribution report** from a ReaxFF **molecular dynamics (MD) simulation**.

It provides a detailed breakdown of the total potential energy into its individual physical components and is the **MD analogue of `fort.58`** (which is written during MM minimization).

---

## Purpose of the `fort.73` file

`fort.73` is used to:

- Analyze **which interaction terms dominate** during MD
- Track how energy components evolve with time
- Verify energy consistency with `fort.71`
- Diagnose instabilities or unphysical behavior in MD runs
- Support force-field development and validation

---

## Output frequency

The output frequency is controlled by the `control`-file keyword:

```
iout1
```

A line is written to `fort.73` every `iout1` MD iterations.

In **Example 3.6**, `iout1 = 5`.

---

## File structure

- One header line listing energy components
- One data line per output step
- Each row corresponds to the same MD iteration reported in `fort.71`

---

## Energy components

All energies are reported in **kcal/mol**.

| Column | Description |
|---|---|
| `Iter` | MD iteration number |
| `Ebond` | Bond energy |
| `Eatom` | Over- and undercoordination energy |
| `Elp` | Lone-pair energy |
| `Emol` | Molecular energy (not used in recent force fields) |
| `Eval` | Valence angle energy |
| `Ecoa` | Valence angle conjugation energy |
| `Ehbo` | Hydrogen bond energy |
| `Etors` | Torsion angle energy |
| `Econj` | Torsion conjugation energy |
| `Evdw` | van der Waals energy |
| `Ecoul` | Coulomb (electrostatic) energy |
| `Echarge` | Charge polarization energy |

---

## Example 3.6: `fort.73` output file

```text
Iter. Ebond Eatom Elp Emol Eval Ecoa Ehbo Etors Econj Evdw Ecoul Echarge

5  -905.00 -7.39 0.00 0.00 0.30 0.00 0.00 4.57 -0.49 255.78 -22.93 11.45
10 -913.27 -6.45 0.00 0.00 0.34 0.00 0.00 4.87 -0.51 262.72 -23.33 11.76
15 -915.74 -5.42 0.00 0.00 0.20 0.00 0.00 5.38 -0.53 263.86 -23.29 11.74
20 -910.80 -5.68 0.00 0.00 0.23 0.00 0.00 5.42 -0.53 258.94 -23.05 11.54
25 -907.76 -6.64 0.00 0.00 0.44 0.00 0.00 4.97 -0.51 257.35 -23.13 11.60
30 -910.68 -6.80 0.00 0.00 0.37 0.00 0.00 4.74 -0.51 260.62 -23.32 11.76
35 -914.36 -6.12 0.00 0.00 0.37 0.00 0.00 4.95 -0.53 263.46 -23.23 11.68
40 -910.62 -6.00 0.00 0.00 0.47 0.00 0.00 5.17 -0.53 259.26 -22.80 11.34
45 -906.77 -6.74 0.00 0.00 0.31 0.00 0.00 4.98 -0.51 256.49 -22.88 11.40
```

---

## Relationship to `fort.71`

- `fort.73` contains **all partial energy terms**, including **charge polarization energy**
- The sum of all energy components in `fort.73` **should equal**:

```
Epot (from fort.71)
```

This makes `fort.73` the **most complete energy decomposition file** produced during MD.

---

## Practical notes

- Sudden jumps in a single energy term often reveal the source of instability
- `Echarge` is especially important in simulations with strong polarization effects
- Commonly plotted alongside `fort.71` for MD diagnostics
- Widely used in automated ReaxKit analysis pipelines

---

## Summary

- `fort.73` provides a **full MD energy decomposition**
- Output frequency controlled by `iout1`
- Energies sum to `Epot` in `fort.71`
- Essential for energy analysis, validation, and force-field development
