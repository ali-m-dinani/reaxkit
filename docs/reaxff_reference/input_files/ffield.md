# `ffield` file — ReaxFF force field definition

The **`ffield`** input file defines all force field parameters used by ReaxFF.

The first line of the `ffield` file contains a **force field identifier** (see [Example 2.8](../images/ffield-input file.png) in the). The remainder of the file is divided into **seven sections**, corresponding to different interaction types.

---

## Structure of the `ffield` file

**Seven sections in order:**

1. General parameters  
2. Atom parameters  
3. Bond parameters  
4. Off-diagonal terms  
5. Valence angle parameters  
6. Torsion angle parameters  
7. Hydrogen bond parameters  

Each section begins with a count of the parameter entries it contains, followed by parameter identifiers and numeric values.

**Figure 1** illustrates the overall layout of the `ffield` file and its sections.

---

## 1. General parameters

The **general parameter section** contains parameters that affect *all* interactions, regardless of atom type.

### Format

- First line:  
  - `npar` — number of general parameters (`i3` format)
- Followed by `npar` lines, each containing:
  - parameter value  
  - parameter identifier

### Important parameters

Two parameters in this section strongly affect performance:

- **Upper Taper radius**  
  Controls the non-bonded cutoff distance.
- **Bond order cutoff**  
  Sets the bond order threshold above which atoms are considered bonded.

> ⚠️ Changing these parameters can significantly speed up ReaxFF but **requires re-parameterization**, as they strongly affect force-field behavior.

---

## 2. Atom parameters

The **atom parameter section** defines element-specific properties.

### Format

1. Number of atom types
2. Four lines of parameter identifiers
3. For each atom type:
   - One line with atom name + 8 parameters (`1x,a2,8f9.4`)
   - Three continuation lines with 8 parameters each (`3x,8f9.4`)

### Notes

- Negative values for bond radii (`cov.r`, `cov.r2`, `cov.r3`) disable the corresponding bond-order contribution.
- Example: if only `cov.r` is positive, only σ-bonds are considered for that atom.

**Table 1** describes the carbon (`C`) atom parameters from Example 2.8.

### Table 1: Carbon atom parameters (`C`)

| Value | Identifier | Description |
|---|---|---|
| 1.3826 | cov.r | σ-bond covalent radius |
| 4.0000 | valency | Valency |
| 12.0000 | a.m. | Atomic mass |
| 2.0195 | Rvdw | van der Waals radius |
| 0.0763 | Evdw | van der Waals dissociation energy |
| 0.8712 | gammaEEM | EEM shielding |
| 1.2360 | cov.r2 | π-bond covalent radius |
| 4.0000 | #el. | Number of valence electrons |
| 10.6359 | alfa | van der Waals parameter |
| 1.9232 | gammavdW | van der Waals shielding |
| 4.0000 | valency | Valency for 1,3-BO correction |
| 40.5154 | Eunder | Undercoordination energy |
| 5.7524 | chiEEM | EEM electronegativity |
| 6.9235 | etaEEM | EEM hardness |
| 1.1663 | cov.r3 | Double π-bond covalent radius |
| 0.0000 | Elp | Lone-pair energy |
| 200.049 | Heat inc. | Heat of formation increment |
| 6.1551 | 13BO1 | Bond order correction |
| 28.6991 | 13BO2 | Bond order correction |
| 12.1086 | 13BO3 | Bond order correction |
| -14.1953 | ov/un | Over-/undercoordination |
| 3.5288 | vval1 | Valence angle energy |
| 6.2998 | vval2 | Valence angle energy |
| 2.9663 | vval3 | Valence angle energy |

*n.u. identifiers are omitted.*

---

## 3. Bond parameters

This section defines bonded interactions between atom pairs.

### Format

1. Number of bond types
2. Two lines of parameter identifiers
3. For each bond type:
   - Line 1: atom-type pair + 8 parameters (`2i3,8f9.4`)
   - Line 2: 8 continuation parameters (`6x,8f9.4`)

> ReaxFF will **terminate immediately** if a bond type appears during a simulation that is not defined in the `ffield` file.

**Table 2** summarizes the carbon–carbon (C–C) bond parameters from Example 2.8.

### Table 2: C–C bond parameters

| Value | Identifier | Description |
|---|---|---|
| 152.0140 | Edis1 | σ-bond dissociation energy |
| 104.0507 | Edis2 | π-bond dissociation energy |
| 72.1693 | Edis3 | Double π-bond dissociation energy |
| 0.2447 | pbe1 | Bond energy |
| -0.7132 | pbo5 | Double π bond order |
| 1.0000 | 13corr | 1,3 bond-order correction |
| 23.5135 | pbo6 | Double π bond order |
| 0.3545 | kov | Overcoordination penalty |
| 0.1152 | pbe2 | Bond energy |
| -0.2069 | pbo3 | π bond order |
| 9.2317 | pbo4 | π bond order |
| -0.1042 | pbo1 | σ bond order |
| 5.9159 | pbo2 | σ bond order |
| 1.0000 | ovcorr | Overcoordination BO correction |

---

## 4. Off-diagonal terms

Off-diagonal parameters override default **combination rules** for:

- Bond order interactions
- van der Waals interactions

### Format

- Number of off-diagonal types
- Parameter identifiers (same line)
- One line per type (`2i3,6f9.4`)

Example: `1 2` defines C–H off-diagonal parameters.

**Table 3** lists the C–H off-diagonal parameters from Example 2.8.

### Table 3: C–H off-diagonal parameters

| Value | Identifier | Description |
|---|---|---|
| 0.0404 | Ediss | vdW dissociation energy |
| 1.8583 | Rvdw | vdW radius |
| 10.3804 | alfa | vdW parameter |
| 1.0376 | cov.r | σ-bond covalent radius |
| -1.0 | cov.r2 | π-bond covalent radius |
| -1.0 | cov.r3 | Double π-bond covalent radius |

Negative values disable π and double-π bond orders.

---

## 5. Valence angle parameters

Defines angle interactions between bonded atom triplets.

### Format

- Number of valence angles
- Parameter identifiers
- One line per angle (`3i3,7f9.4`)

Angles not defined in the `ffield` file are **ignored**.

**Table 4** describes the C–C–C valence angle parameters.

### Table 4: C–C–C valence angle parameters

| Value | Identifier | Description |
|---|---|---|
| 70.2140 | Thetao | 180° − equilibrium angle |
| 14.0458 | ka | First force constant |
| 2.0508 | kb | Second force constant |
| 0.0000 | pconj | Valence conjugation |
| 0.0000 | pv2 | Undercoordination |
| 35.9933 | kpenal | Penalty energy |
| 1.0400 | pv3 | Energy/bond order |

*This corresponds to an equilibrium angle of 109.786° for C–C–C σ bonds.*

---

## 6. Torsion angle parameters

Defines dihedral interactions.

### Identification modes

1. **Central-bond based** (e.g. `0 1 1 0` → all C–C torsions)
2. **Four-atom specific** (e.g. `1 1 1 2` → C–C–C–H)

Four-atom identifiers override central-bond definitions.

**Table 5** lists C–C–C–C torsion parameters from Example 2.8.

### Table 5: C–C–C–C torsion parameters

| Value | Identifier | Description |
|---|---|---|
| 0.0000 | V1 | V1 torsion barrier |
| 28.8256 | V2 | V2 torsion barrier |
| 0.1796 | V3 | V3 torsion barrier |
| -4.6957 | V2(BO) | V2 bond-order dependence |
| -1.3130 | vconj | Torsion conjugation |

Torsions not defined in the `ffield` file are ignored.

---

## 7. Hydrogen bond parameters

Defines hydrogen bond interactions.

### Format

- Number of hydrogen bond types
- Parameter identifiers
- One line per hydrogen bond (`3i3,4f9.4`)

Example identifier `1 2 1` corresponds to a **C–H···C** hydrogen bond.

**Table 6** summarizes C–H···C hydrogen bond parameters.

### Table 6: C–H···C hydrogen bond parameters

| Value | Identifier | Description |
|---|---|---|
| 2.0347 | Rhb | Equilibrium H-bond distance |
| 0.0000 | Dehb | Dissociation energy |
| 4.9076 | vhb1 | H-bond / bond-order term |
| 4.2357 | vhb2 | H-bond parameter |

Hydrogen bonds not defined in the `ffield` file are ignored.
