# `geo` file — System geometry input for ReaxFF

The **`geo`** file describes the system geometry used as input to **ReaxFF**.

ReaxFF currently supports these geometry input formats:

- `.geo`
- `.bgf`
- `.xyz`
- z-matrix

By default, ReaxFF assumes `geo` contains either **`.geo`** or **z-matrix** format. To use other formats, set `igeofo` in the **`control`** file as follows:

- `igeofo = 1` → `.bgf` format
- `igeofo = 2` → `.xyz` format

> **Note:** Future development is primarily centered around **`.bgf`** and moves away from **`.geo`**. For that reason, only a brief description of `.geo` is provided here.

All `geo` input formats are **format sensitive**.

---

## Quick navigation

- [Supported formats](#supported-formats)
- [BGF format](#bgf-format)
  - [Non-periodic example](#example-21-non-periodic-bgf-input-file)
  - [Periodic example](#example-22-periodic-bgf-input-geometry-file)
  - [Restraints example](#example-23-bgf-input-file-with-restraints)
  - [`RUTYPE` options](#table-22-rutype-keywords-supported-by-reaxff)
  - [`VCHANGE` example](#example-24-periodic-bgf-input-followed-by-vchange)
- [GEO format](#geo-format)
- [Z-matrix format](#z-matrix-format)
- [XYZ format](#xyz-format)

---

## Supported formats

ReaxFF supports these geometry input formats in the `geo` input:

1. **`.bgf`** (recommended direction for future support)
2. **`.geo`**
3. **z-matrix**
4. **`.xyz`**

The active format is selected by `igeofo` in the `control` file (see top of this page).

---

## BGF format

The `.bgf` format is **keyword-driven**: each line starts with a keyword followed by data associated with that keyword.

This format is used by multiple molecular simulation tools (e.g., Cerius2, Jaguar). Applications typically ignore lines starting with unrecognized keywords, which improves portability.

### Keywords recognized by ReaxFF

- `BIOGRF [VERSION]` / `XTLGRF [VERSION]`: `.bgf` version marker (ReaxFF expects **200**)
- `DESCRP [NAME]`: system description; can be used in `trainset.in`
- `REMARK`: remarks (multiple lines allowed)
- `RUTYPE [KEYWORD ...]`: run parameters (see [Table 1](#table-22-rutype-keywords-supported-by-reaxff))
- `FORMAT [STRING]`: format metadata (**ignored by ReaxFF**; cannot change parsing)
- `HETATM [...]`: atom definition (type + Cartesian coordinates)
- `CONECT [...]`: connection table (**ignored by ReaxFF**; ReaxFF computes its own connections)
- `CRYSTX [A B C Alpha Beta Gamma]`: periodic cell lengths (Å) and angles (deg)
- `END`: end of one geometry

Lines starting with `#` are ignored and can be used as comments.

> **Coordinate convention:** ReaxFF uses **Cartesian** coordinates (not fractional) for atom positions.

---

### Example 2.1: Non-periodic `.bgf` input file

**Figure 1** shows a basic, non-periodic `.bgf` geometry input.

```text
BIOGRF 200
DESCRP Ethane_radical.
REMARK Example
RUTYPE NORMAL RUN

# THIS LINE IS IGNORED

FORMAT ATOM
(a6,1x,i5,1x,a5,1x,a3,1x,a1,1x,a5,3f10.5,1x,a5,i3,i2,1x,f8.5)

HETATM 1 C 39.53649 39.80304 39.57992 C 1 1 -0.2894
HETATM 2 H 39.96404 38.93497 39.07954 H 1 1 0.1031
HETATM 3 C 40.55862 40.34907 40.60075 C 1 1 -0.2330
HETATM 4 H 39.30695 40.55630 38.82721 H 1 1 0.1048
HETATM 5 H 38.62048 39.49467 40.08241 H 1 1 0.1048
HETATM 6 H 40.65314 39.88418 41.56157 H 1 1 0.1048
HETATM 7 H 41.36027 40.97776 40.26860 H 1 1 0.1048

FORMAT CONECT (a6,12i6)
CONECT 1 2 3 4 5
CONECT 2 1
CONECT 3 1 6 7
CONECT 4 1
CONECT 5 1
CONECT 6 3
CONECT 7 3

END
```

**Notes**
- `FORMAT ...` lines are present for compatibility, but are **ignored** by ReaxFF.
- `HETATM` includes an atom number, atom type, `x y z` in Å, force-field type, two unused switches, and an (unused) partial charge.
- `CONECT` is **ignored** by ReaxFF; it computes connectivity internally.

---

## Table 2.2: `RUTYPE` keywords supported by ReaxFF

`RUTYPE` selects whether ReaxFF uses run switches from the `control` file or overrides them with run-specific options.

If `RUTYPE` is `NORMAL RUN` (as in Example 2.1), ReaxFF uses switches defined in the `control` file. Otherwise, it can override certain switches using the keywords below.

| Keyword | Description |
|---|---|
| `NORMAL RUN` | Use switches in `control` file |
| `MAXIT [NUMBER]` | Stop MM run after `NUMBER` iterations |
| `ENDPO [NUMBER]` | Stop MM run when `RMSG` drops below `NUMBER` |
| `MAXMOV [NUMBER]` | Maximum atom movement (in 10^-6 Å) during steepest-descent MM minimization. With `NUMBER = 0`, conjugate gradient is used. |
| `SINGLE POINT` | Stop MM run after first point |
| `DOUBLE PRECISION` | Double MM maximum iterations and halve `RMSG` end criterion (vs. `control` file) |
| `LOW PRECISION` | Halve MM maximum iterations (vs. `control` file) |
| `CELL OPT [NUMBER]` | Perform numerical cell optimization in MM (see `control` options for `NUMBER`) |
| `NO CELL OPT` | Do not perform numerical cell optimization |

---

### Example 2.2: Periodic `.bgf` input geometry file

This example defines periodic cell parameters using `CRYSTX` and indicates a periodic system using `XTLGRF`.

```text
XTLGRF 200
DESCRP fcc_1
REMARK Platinum fcc-structure
RUTYPE NORMAL RUN

CRYSTX 4.50640 4.50640 4.50640 90.00000 90.00000 90.00000

FORMAT ATOM
(a6,1x,i5,1x,a5,1x,a3,1x,a1,1x,a5,3f10.5,1x,a5,i3,i2,1x,f8.5)

HETATM 1 Pt 0.00000 0.00000 0.00000 Pt 1 1 0.00000
HETATM 2 Pt 2.25138 2.25138 0.00000 Pt 1 1 0.00000
HETATM 3 Pt 2.25138 0.00000 2.25138 Pt 1 1 0.00000
HETATM 4 Pt 0.00000 2.25138 2.25138 Pt 1 1 0.00000

FORMAT CONECT (a6,12i6)
END
```

**Important behavior**
- ReaxFF treats systems as periodic in principle.
- For *non-periodic* systems, it uses large default cell parameters set by `axis1`, `axis2`, `axis3` in the `control` file.
- If cell parameters are provided in the `geo` file (via `CRYSTX`), the `control` file cell parameters are ignored.

---

### Example 2.3: `.bgf` input file with restraints

The `.bgf` format can include restraints used during MM or MD to drive reactions or enforce conformational changes.

```text
BIOGRF 200
DESCRP Hshift11
RUTYPE NORMAL RUN

FORMAT BOND RESTRAINT (15x,2i4,f8.4,f8.2,f8.5,f10.7)
# At1 At2 R12 Force1 Force2 dR12/dIteration(MD only)
BOND RESTRAINT 1 2 1.0900 7500.00 0.25000 0.0000000

FORMAT ANGLE RESTRAINT (16x,3i4,2f8.2,f8.4,f9.6)
# At1 At2 At3 Angle Force1 Force2 dAngle/dIteration (MD)
ANGLE RESTRAINT 1 2 3 120.00 250.00 1.00000 0.0000

FORMAT TORSION RESTRAINT (18x,4i4,2f8.2,f8.4,f9.6)
# At1 At2 At3 At4 Angle Force1 Force2 dAngle/dIt
TORSION RESTRAINT 1 2 3 4 45.00 250.00 1.00000 0.0000

FORMAT MASCEN RESTRAINT FREE FORMAT
# x/y/z At1 At2 R At3 At4 Force1 Force2
MASCEN RESTRAINT x 1 3 1.50 4 7 50.00 0.25

FORMAT ATOM
(a6,1x,i5,1x,a5,1x,a3,1x,a1,1x,a5,3f10.5,1x,a5,i3,i2,1x,f8.5)

HETATM 1 C 39.53692 39.80281 39.57996 C 1 1 0.00000
HETATM 2 H 39.96200 38.93424 39.07781 H 1 1 0.00000
HETATM 3 C 40.55717 40.34771 40.59881 C 1 1 0.00000
HETATM 4 H 39.30845 40.55556 38.82947 H 1 1 0.00000
HETATM 5 H 38.62310 39.49566 40.08262 H 1 1 0.00000
HETATM 6 H 40.65332 39.88631 41.56086 H 1 1 0.00000
HETATM 7 H 41.35903 40.97771 40.27048 H 1 1 0.00000

FORMAT CONECT (a6,12i6)
END
```

#### Restraint keywords

##### Bond restraint

`BOND RESTRAINT [At1 At2 R12 Force1 Force2 dR12/dIteration]`

Adds an additional restraint energy (see **Equation 1**) aiming to keep the distance between atoms `At1` and `At2` near `R12`.

**Equation 1 (restraint energy)**

``
E_restraint = Force_1*(1 - exp(Force_2*(R_ij-R12)^2))
``

During **MD**, `R12` can be updated each iteration by `dR12/dIteration` (reaction driving). This is **not available** during MM minimization.

##### Angle restraint

`ANGLE RESTRAINT [At1 At2 At3 Angle Force1 Force2 dAngle/dIteration]`

Restrains the angle defined by atoms `At1–At2–At3` (independent of connectivity). Can be driven during MD by `dAngle/dIteration`.

##### Torsion restraint

`TORSION RESTRAINT [At1 At2 At3 At4 Angle Force1 Force2 dAngle/dIteration]`

Restrains torsion angle `At1–At2–At3–At4`. Currently, this restraint **only** works between **connected** atoms, and **At2 should be smaller than At3**. Driving through `0°` or `180°` may cause problems.

##### Center-of-mass restraint

`MASCEN RESTRAINT [x/y/z At1 At2 R At3 At4 Force1 Force2]`

Restrains the center-of-mass of atoms `At1..At2` to be a distance `R` away from the center-of-mass of atoms `At3..At4` in the specified axis direction (`x`, `y`, or `z`).

---

### Multiple geometries per `geo` file

ReaxFF can run multiple simulations from one `geo` file by concatenating geometries and separating them with **one empty line after each `END`**.

Alternatively, use `models.in` to provide paths to multiple `geo` files.

---

### Example 2.4: Periodic `.bgf` input followed by `VCHANGE`

`VCHANGE [NUMBER]` repeats the previous simulation with **rescaled cell volume and coordinates**. The rescaling factor is `NUMBER * 100%` (e.g., `0.80` → 80% volume).

ReaxFF automatically uses `RUTYPE NO CELL OPT` (see [Table 1](#table-22-rutype-keywords-supported-by-reaxff)) for structures specified with `VCHANGE`.

```text
XTLGRF 200
DESCRP fcc_1
REMARK Platinum fcc-structure
RUTYPE SINGLE POINT

CRYSTX 4.50640 4.50640 4.50640 90.00000 90.00000 90.00000

FORMAT ATOM
(a6,1x,i5,1x,a5,1x,a3,1x,a1,1x,a5,3f10.5,1x,a5,i3,i2,1x,f8.5)

HETATM 1 Pt 0.00000 0.00000 0.00000 Pt 1 1 0.00000
HETATM 2 Pt 2.25138 2.25138 0.00000 Pt 1 1 0.00000
HETATM 3 Pt 2.25138 0.00000 2.25138 Pt 1 1 0.00000
HETATM 4 Pt 0.00000 2.25138 2.25138 Pt 1 1 0.00000

FORMAT CONECT (a6,12i6)
END

XTLGRF 200
DESCRP fcc_2
REMARK Rerun fcc_1 at 80% volume

VCHANGE 0.80
END
```

---

## GEO format

As an alternative to `.bgf`, ReaxFF supports `.geo` input. The first line contains a **control character** and structure name. The following lines list atom number, atom type, and Cartesian coordinates.

### Example 2.5: `.geo` input file

```text
C Ethyl_radical
1 C 0.39536921883140E+02 0.39802812097390E+02 0.39579964797377E+02
2 H 0.39962000508882E+02 0.38934237894502E+02 0.39077807973956E+02
3 C 0.40557168455097E+02 0.40347711311539E+02 0.40598809008712E+02
4 H 0.39308447421804E+02 0.40555557250666E+02 0.38829468433362E+02
5 H 0.38623101361336E+02 0.39495660102816E+02 0.40082615760111E+02
6 H 0.40653318128573E+02 0.39886313307789E+02 0.41560863579234E+02
7 H 0.41359032241168E+02 0.40977708035298E+02 0.40270480449231E+02
```

### Table 2.3: Control-character options in `.geo` format

| Control character | Effect |
|---|---|
| `C` | Normal run as defined in `control` file |
| `F` | Use cell parameters (defined on lines 2 and 3 of `.geo`) |
| `1` | Single point |
| `D` | Double precision run |
| `H` | Low precision run |
| `5` | Use cell parameters; single point |

> More detailed `.geo` options may be available from the original author; ReaxFF development is moving away from `.geo` toward `.bgf`.

---

## Z-matrix format

The z-matrix format uses internal coordinates, which is convenient for building molecules.

### Example 2.6: z-matrix input file

```text
I Ethyl_radical
1 C
1 2 C 1.08962
1 2 3 C 42.38253 2.15999
2 3 1 4 H -119.14400 110.52474 1.08723
2 3 1 5 H 119.14417 110.52466 1.08723
5 1 3 6 H 38.20847 120.04903 1.07130
5 1 3 7 H -159.92026 120.04917 1.07130
```

**Format notes**
- The `I` in position 3 on the first line is required as a format identifier, followed by the structure name.
- Lines use the `4i3,1x,a2,3f10.5` format and contain:
  - `atl, atk, atj, ati, atype, torsijkl, angleijk, Rij`
- `torsijkl`: torsion angle for `ati–atj–atk–atl`
- `angleijk`: angle for `ati–atj–atk`
- `Rij`: distance for `ati–atj`
- `atype`: atom type for `ati`

**Limitation**
- z-matrix format cannot specify cell parameters; it must use default `axis1`, `axis2`, `axis3` from the `control` file.

---

## XYZ format

The `.xyz` format is widely used due to its simplicity and compatibility with viewers/tools (e.g., Icarus, Molden, Xmol).

### Example 2.7: `.xyz` input file

```text
7
Ethyl_radical
C 39.53692 39.80281 39.57996
H 39.96200 38.93424 39.07781
C 40.55717 40.34771 40.59881
H 39.30845 40.55556 38.82947
H 38.62310 39.49566 40.08262
H 40.65332 39.88631 41.56086
H 41.35903 40.97771 40.27048
```

**Format notes**
- Line 1: number of atoms
- Line 2: structure name
- Remaining lines: `type x y z` (Cartesian)

**Limitation**
- `.xyz` cannot specify cell parameters; it must use default `axis1`, `axis2`, `axis3` from the `control` file.
