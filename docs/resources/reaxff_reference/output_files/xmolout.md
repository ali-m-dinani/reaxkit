# `xmolout` file — Atomic trajectory output

The **`xmolout`** file contains the **atomic trajectory** generated during a ReaxFF **MD run** or **MM minimization**.  
It is generally the **most useful output file** for visualization and post‑processing.

Many molecular visualization programs can read `xmolout` directly, including:
- **Icarus**
- **Xmol**
- **Molden**
- **Jmol**

These programs can display **animated trajectories** of the simulation.

---

## Purpose of the `xmolout` file

- Store atomic positions over time
- Visualize molecular motion and reactions
- Analyze trajectories frame‑by‑frame
- Interface ReaxFF with external viewers and analysis tools

---

## Output behavior

### MD simulations
- Frame output frequency is controlled by the **`iout2`** keyword in the `control` file
- One frame is written every `iout2` MD iterations

### MM minimizations
- Frame output frequency is controlled by **`iout4`** in the `control` file

---

## Appending behavior

- `xmolout` is **always appended**, never overwritten
- If a previous `xmolout` exists in the run directory, new frames are added to the end
- The `exe` script may delete old `xmolout` files before a run
- If not deleted, trajectories from **multiple runs or geometries** accumulate

When running ReaxFF on:
- multiple geometries in a single `geo` file, or
- multiple geometries listed in `models.in`

the `xmolout` file will contain **frames from all geometries** in sequence.

---

## File format

The `xmolout` file follows an **extended XYZ-style format**.

Each frame consists of:

1. Number of atoms  
2. Structure identifier  
3. One line per atom: atom type and Cartesian coordinates  

### Per‑frame layout

```
<N_atoms>
<Structure_identifier>
<Atom>  x  y  z
...
```

Coordinates are in **Ångström**.

---

## Example 3.1: `xmolout` file

The following example shows **three frames** from an MD simulation of an **ethyl radical**.

```text
7
Ethyl_radical
C 39.19924 40.08333 39.66585
H 40.88239 40.79608 40.81249
C 40.61413 39.91227 40.24250
H 38.86032 41.06389 39.38266
H 38.59716 39.25081 39.41763
H 41.33155 39.78117 39.45850
H 40.65892 39.04746 40.86980

7
Ethyl_radical
C 39.20278 40.09024 39.65697
H 40.80762 40.70179 40.96412
C 40.61564 39.90857 40.24468
H 38.93202 40.97423 39.12611
H 38.47436 39.29665 39.73567
H 41.35468 39.98863 39.43895
H 40.70164 38.93982 40.75604

7
Ethyl_radical
C 39.20278 40.09024 39.65697
H 40.80762 40.70179 40.96412
C 40.61564 39.90857 40.24468
H 38.93202 40.97423 39.12611
H 38.47436 39.29665 39.73567
H 41.35468 39.98863 39.43895
H 40.70164 38.93982 40.75604
```

---

## Optional extended output (`ixmolo`)

If the **`ixmolo`** keyword in the `control` file is set to `1`, additional information is written to `xmolout`, including:

- Atomic velocities
- Molecule numbers

Higher `ixmolo` values may include:
- Strain energy per atom
- Total bond order per atom

(See the `control` file documentation for details.)

---

## Practical notes

- `xmolout` is the **primary trajectory file** for ReaxFF
- Ideal for visualization, diffusion analysis, and reaction tracking
- Can grow very large for long simulations — periodic cleanup may be needed
- Compatible with most XYZ‑based analysis tools

---

## Summary

- `xmolout` stores ReaxFF trajectories
- Written during MD and MM runs
- Appended continuously
- Directly readable by common molecular viewers
- Central to visualization and trajectory analysis workflows
