# `koppel2` file — Linking force field parameters during optimization

The **`koppel2`** file allows users to **link multiple force field parameters** during a ReaxFF **force field optimization**.

Linked parameters are constrained to **retain identical values** throughout the optimization procedure. This is useful when different parameter entries are physically equivalent or should remain consistent by design.

---

## Purpose of the `koppel2` file

During optimization, ReaxFF normally treats each parameter listed in `params` independently. The `koppel2` file overrides this behavior by:

- Defining **groups of linked parameters**
- Enforcing **identical values** for all parameters in each group
- Reducing the dimensionality of the optimization problem
- Improving stability and physical consistency

---

## File handling

- The `koppel2` file is copied by the `exe` script to **`fort.23`**
- ReaxFF reads parameter-linking instructions exclusively from `fort.23`
- The filename `koppel2` itself is not read directly by ReaxFF

---

## File format

The `koppel2` file consists of **link blocks**.

Each block contains:
1. A **reference parameter identifier**
2. The **number of parameters linked** to it
3. A list of the linked parameter identifiers

### Identifier format

Parameter identifiers follow the same convention as in the `params` file:

```
<section> <type> <parameter>
```

Using fixed-width formatting:
```
4i3
```

---

## Example 2.17: `koppel2` file

```text
5 1 7 5   ! Reference parameter; number of links

5 2 7
<!-- -->
5 3 7
<!-- -->
5 4 7    ! Parameters linked to parameter 5 1 7
<!-- -->
5 5 7
5 6 7
```

---

## Interpretation of the example

The first line defines the **reference parameter**:

```
5 1 7
```

This corresponds to:
- **Section**: 5 → valence angle parameters
- **Type**: 1
- **Parameter**: 7

The second integer (`5`) indicates that **five additional parameters** are linked to this reference.

Each subsequent line defines one linked parameter:

- `5 2 7`
- `5 3 7`
- `5 4 7`
- `5 5 7`
- `5 6 7`

All six parameters (the reference plus five linked parameters) are forced to **share the same value** during optimization.

---

## Typical use cases

- Enforcing symmetry between chemically equivalent interactions
- Linking parameters across multiple atom types
- Reducing overfitting in force field optimization
- Maintaining physical consistency in large parameter sets

---

## Interaction with other optimization files

- **`params`** defines *which* parameters are optimized and how
- **`koppel2`** defines *which optimized parameters are linked*
- **`trainset.in`** defines *what data is fitted*
- **`control`** controls optimization behavior (`parsca`, `parext`)

Together, these files define the complete ReaxFF force field fitting workflow.

---

## Summary

- `koppel2` links force field parameters during optimization
- All linked parameters retain identical values
- Uses the same parameter identifier convention as `params`
- Copied to `fort.23` and read by ReaxFF
- Essential for constrained and physically consistent force field fitting
