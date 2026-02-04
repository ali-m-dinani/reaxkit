# `fort.7` / `fort.8` files — Connectivity and charge output

The **`fort.7`** and **`fort.8`** files contain ReaxFF‑generated **connectivity tables** and **partial charge information**.  
They describe which atoms are bonded, the corresponding **bond orders**, molecular membership, and atomic charges.

These files are central for:
- Bond analysis
- Molecule identification
- Charge tracking
- Graph‑based post‑processing (e.g. fragmentation, reactions)

---

## Difference between `fort.7` and `fort.8`

| File | Contents |
|---|---|
| **`fort.8`** | All bonds and bond orders |
| **`fort.7`** | Only bonds with bond order > `cutof2` (from `control` file) |

`fort.7` is therefore a **filtered version** of `fort.8`, typically used for molecular graphs.

---

## Output frequency

Both files are updated at the same frequency as `xmolout`:

- **MD simulations**: every `iout2` iterations  
- **MM runs**: every `iout4` iterations  

By default, these files are **overwritten** with the most recent data.

Setting the `control` keyword:
```
iappen = 1
```
causes both `fort.7` and `fort.8` to be **appended** instead.

---

## File structure

### Header line

The first line contains:
```
<number_of_atoms> <structure_identifier>
```

Example:
```
7 Ethyl_radical
```

---

### Atom records

Each subsequent line corresponds to **one atom** and contains:

1. Atom number  
2. Atom type number (as defined in `ffield`)  
3. Connectivity list (bonded atom indices)  
4. Molecule number  
5. Bond orders (one per bonded neighbor)  
6. Sum of bond orders  
7. Number of lone pairs  
8. Partial charge  

---

## Example 3.2: `fort.7` connection table

```text
7 Ethyl_radical

1 1 2 3 4 5 0 1 0.985 1.024 0.986 0.986 0.000 3.981 0.000 -0.289
2 2 1 0 0 0 0 1 0.985 0.000 0.000 0.000 0.000 0.985 0.000 0.103
3 1 1 6 7 0 0 1 1.024 0.987 0.987 0.000 0.000 2.999 0.000 -0.233
4 2 1 0 0 0 0 1 0.986 0.000 0.000 0.000 0.000 0.986 0.000 0.105
5 2 1 0 0 0 0 1 0.986 0.000 0.000 0.000 0.000 0.986 0.000 0.105
6 2 3 0 0 0 0 1 0.987 0.000 0.000 0.000 0.000 0.987 0.000 0.105
7 2 3 0 0 0 0 1 0.987 0.000 0.000 0.000 0.000 0.987 0.000 0.105
```

---

## Column interpretation (conceptual)

| Field | Meaning |
|---|---|
| Atom number | Index of atom |
| Atom type | Force‑field atom type |
| Connectivity | Indices of bonded atoms |
| Molecule number | Molecular fragment ID |
| Bond orders | Bond order per connection |
| Σ bond orders | Total bond order |
| Lone pairs | Number of lone pairs |
| Partial charge | Atomic charge |

The **number of connectivity entries and bond orders varies** depending on how many neighbors an atom has.

---

## Practical notes

- `fort.7` is commonly used for **molecule detection** and **reaction tracking**
- `fort.8` is useful for **detailed bond‑order analysis**
- Both files are tightly coupled to `cutof2`, `iout2`, `iout4`, and `iappen`
- They are ideal inputs for graph‑based post‑processing tools

---

## Summary

- `fort.7` / `fort.8` store ReaxFF connectivity and charge data
- Updated synchronously with `xmolout`
- `fort.7` is a bond‑order‑filtered subset of `fort.8`
- Essential for bonding, molecule, and charge analysis workflows
