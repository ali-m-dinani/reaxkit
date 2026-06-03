# `addmol.bgf` file — On-the-fly molecule insertion (GCMD)

The **`addmol.bgf`** file is an **optional input** that allows ReaxFF to **insert molecules during an MD simulation** at user-defined intervals.

This enables a **Grand-Canonical Molecular Dynamics (GCMD)**–like workflow, where molecules are periodically added with controlled temperature or velocity, typically at random locations in the simulation box.

---

## Purpose of `addmol.bgf`

Using `addmol.bgf`, ReaxFF can:

- Insert molecules during an ongoing MD simulation
- Control **insertion frequency**
- Specify **initial velocities or temperatures**
- Enforce **minimum distance constraints** to avoid overlaps
- Perform gas–surface interaction or adsorption simulations

During execution, ReaxFF prints diagnostic messages to the terminal describing molecule placement attempts and outcomes.

---

## Example 2.13: `addmol.bgf` file (O₂ insertion)

The following example inserts an **O₂ molecule** every **1000 MD steps** at **250 K**, at a random position in the simulation box.

```text
BIOGRF 200
DESCRP O2

FREQADD 1000
VELADD 1

STARTX -9000.0
STARTY -9000.0
STARTZ -9000.0

ADDIST 3.0
NATTEMPT 050
TADDMOL 250.0

FORMAT ATOM
(a6,1x,i5,1x,a5,1x,a3,1x,a1,1x,a5,3f10.5,1x,a5,i3,i2,1x,f8.5)

HETATM 1 O 0.00000 0.00000 0.00000 O 1 3 0.00000
HETATM 2 O 1.24500 0.00000 0.00000 O 1 3 0.00000

END
```

---

## Keyword reference

### `FREQADD`
Insertion frequency (in MD iterations).

- The **first insertion** always occurs at iteration **5**
- Subsequent insertions occur every `FREQADD` steps

**Example**
- `FREQADD 1000` → insert at 5, 1000, 2000, 3000, ...

---

### `VELADD`
Controls how initial velocities are assigned.

| Value | Meaning |
|---|---|
| `1` | Assign random velocities based on `TADDMOL` |
| `2` | Read velocities from `addmol.vel` |

---

### `STARTX`, `STARTY`, `STARTZ`
Coordinates of the **center of mass** of the inserted molecule.

- If value `< -5000.0` → random position
- Otherwise → fixed position

---

### `ADDIST`
Minimum allowed distance (Å) between the inserted molecule and existing atoms.

Used to prevent atomic overlap during insertion.

---

### `NATTEMPT`
Maximum number of placement attempts.

If no valid position is found after `NATTEMPT` tries (respecting `ADDIST`), the insertion is skipped.

---

### `TADDMOL`
Temperature (K) of the inserted molecule.

- Only used when `VELADD = 1`
- Ignored when velocities are read from `addmol.vel`

---

## Molecule definition

The molecular structure is defined using standard **`.bgf` atom records**:

- `FORMAT ATOM` lines describe the expected atom record layout
- `HETATM` lines define atomic positions and types
- `CONECT` records are not required
- Geometry is inserted as a rigid molecule at placement

---

## Example 2.14: `addmol.vel` file

When `VELADD = 2`, velocities are read from a separate **`addmol.vel`** file.

This file has the **same format** as velocity sections in the `vels` restart file.

```text
Atom velocities (Angstrom/s):

0.676920600871422E+13  0.250389491659681E+13   0.385204179579294E+03
0.638733784812228E+13  0.125025292253893E+13  -0.372288199177400E+03
```

Each line corresponds to the velocity vector of one atom in the inserted molecule.

---

## Output behavior

- ReaxFF writes **placement diagnostics** to the terminal
- Failed placement attempts and skipped insertions are reported
- Successful insertions are reflected in trajectory and energy outputs

---

## Typical use cases

- Gas-phase molecule injection
- Surface adsorption studies
- Reactive flux simulations
- Modeling open-system environments

The `addmol.bgf` mechanism provides a flexible and powerful way to extend ReaxFF MD simulations beyond fixed-particle ensembles.
