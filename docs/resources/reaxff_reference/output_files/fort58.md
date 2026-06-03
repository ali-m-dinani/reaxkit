# `fort.58` file — MM partial energy decomposition

The **`fort.58`** file contains a **partial energy breakdown** from a molecular mechanics (MM) energy minimization in ReaxFF.

It complements the total-energy report in **`fort.57`** by decomposing the potential energy into individual physical contributions (bonding, angles, van der Waals, Coulomb, etc.).

---

## Purpose of the `fort.58` file

The `fort.58` output is used to:

- Analyze **which interaction terms dominate** the energy
- Diagnose problematic force-field components
- Understand convergence behavior during MM minimization
- Support force-field development and debugging

---

## When `fort.58` is generated

- Generated during **MM energy minimization**
- Written once per MM iteration
- Corresponds directly to iterations reported in `fort.57`
- May be removed automatically if `iout5 = 1` in the `control` file

---

## File structure

The file consists of:

1. A **structure identifier**
2. A **header line** listing energy components
3. One line per MM iteration with partial energies

---

## Energy components

Each iteration reports the following energy terms (in **kcal/mol**):

| Column | Description |
|---|---|
| `Eatom` | Over- and undercoordination energy |
| `Elopa` | Lone-pair energy |
| `Ebond` | Bond energy |
| `Emol` | Molecular energy (not used in current force fields) |
| `Eval` | Valence angle energy |
| `Ecoa` | Valence angle conjugation energy |
| `Ehb` | Hydrogen bond energy |
| `Etor` | Torsion angle energy |
| `Econj` | Torsion conjugation energy |
| `Evdw` | van der Waals energy |
| `Ecoul` | Coulomb (electrostatic) energy |

---

## Example 3.4: `fort.58` output file

```text
Ethyl_radical

Iter. Eatom Elopa Ebond Emol Eval Ecoa Ehb Etor Econj Evdw Ecoul

0 -6.922 0.000 -907.050 0.000 0.237 0.000 0.000 4.715 -0.508 257.018 -22.970
1 -6.741 0.000 -907.442 0.000 0.244 0.000 0.000 4.801 -0.512 257.123 -22.966
2 -6.559 0.000 -908.118 0.000 0.242 0.000 0.000 4.876 -0.516 257.524 -22.965
3 -6.196 0.000 -909.827 0.000 0.233 0.000 0.000 5.013 -0.523 258.720 -22.972
4 -6.283 0.000 -910.220 0.000 0.230 0.000 0.000 4.954 -0.521 259.253 -22.985
5 -6.305 0.000 -910.733 0.000 0.234 0.000 0.000 4.924 -0.520 259.819 -22.996
6 -6.340 0.000 -911.121 0.000 0.234 0.000 0.000 4.896 -0.519 260.277 -23.006
```

---

## Relationship to `fort.57`

- `fort.57` reports the **total potential energy** (`Epot`) and convergence metrics
- `fort.58` reports **partial energy components**
- The sum of energies in `fort.58` **does not equal** `Epot` in `fort.57`

### Why the totals differ

The energy required to generate atomic charges (**`Echarge`**) is **not included** in `fort.58`.

This missing contribution is reported separately in:
- **`fort.73`** / `energylog`

As a result:
```
Σ(Energies in fort.58) ≠ Epot (fort.57)
```

---

## Practical notes

- Large changes in a single term (e.g. `Ebond` or `Evdw`) often indicate the source of instability
- Useful for tuning force-field parameters during optimization
- Frequently parsed together with `fort.57` in automated workflows

---

## Summary

- `fort.58` provides a **decomposed MM energy report**
- Written during MM minimization
- Complements `fort.57` but does not include charge-generation energy
- Essential for detailed force-field diagnostics
