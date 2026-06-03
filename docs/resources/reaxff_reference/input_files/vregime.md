# `vregime.in` file — Cell parameter manipulation during MD

The **`vregime.in`** file is an **optional input** that allows controlled manipulation of **cell parameters** during an MD simulation.

Its structure and usage are closely analogous to the `tregime.in` file, but instead of temperatures it defines **volume, lattice, or angle deformation regimes**. This makes it especially useful for **mechanical loading**, **strain-controlled simulations**, and **density equilibration** of condensed-phase systems.

When `vregime.in` is provided, ReaxFF generates an additional output file:

- **`fort.77`** — records the targeted and applied cell-parameter changes over time

---

## Purpose of `vregime.in`

Typical applications include:

- Applying uniaxial or multiaxial strain
- Crack propagation simulations
- Mechanical response studies
- Driving amorphous materials toward correct density
- Controlled lattice expansion or compression

---

## General properties

- **Format-free**: spacing and column alignment are flexible
- **Stage-based**: each line defines a new deformation stage
- **Multi-parameter**: multiple cell parameters can be modified simultaneously
- **Optional rescaling**: atomic coordinates may or may not follow the cell change

---

## File format

Each non-comment line defines **one volume (cell) regime stage**.

Comment lines begin with `#`.

### Column definition (conceptual)

```
start  #V  type1  change/it  rescale  [type2  change/it  rescale] ...
```

Where:

- **start** — MD iteration at which this regime begins  
- **#V** — number of cell-parameter modifications in this stage  
- **type** — cell parameter to modify (`a`, `b`, `c`, `alfa`, `beta`, `gamma`)  
- **change/it** — incremental change applied per MD iteration  
- **rescale** — whether atomic coordinates are rescaled (`y` or `n`)  

Additional parameter blocks may follow on the same line.

---

## Supported cell parameters

| Keyword | Meaning |
|---|---|
| `a` | Cell length *a* |
| `b` | Cell length *b* |
| `c` | Cell length *c* |
| `alfa` | Cell angle α |
| `beta` | Cell angle β |
| `gamma` | Cell angle γ |

---

## Example 2.17: `vregime.in` input file

The following example applies staged lattice and angle deformations.

```text
# Volume regimes
# start  #V  type1  change/it  rescale   type2  change/it  rescale

0000     2   alfa   0.050000   y         beta  -0.05      y
0100     2   beta   0.050000   y         alfa  -0.05      y
0200     2   a      0.010000   y         b     -0.010     y
0300     2   a     -0.010000   y         b      0.010     y
0400     4   a     -0.010000   y         alfa   0.050     y        b   0.01   y   beta 0.05 y
```

---

## Interpretation of the example

### Stage 1 (iteration 0 →)
- Increase **α** angle
- Decrease **β** angle
- Atomic coordinates are rescaled with the cell

### Stage 2 (iteration 100 →)
- Increase **β**
- Decrease **α**
- Coordinates rescaled

### Stage 3 (iteration 200 →)
- Increase lattice constant **a**
- Decrease lattice constant **b**
- Coordinates rescaled

### Stage 4 (iteration 300 →)
- Reverse the previous **a/b** strain

### Stage 5 (iteration 400 →)
- Apply simultaneous strain to:
  - **a**
  - **b**
  - **α**
  - **β**
- Coordinates rescaled for all changes

---

## Rescaling behavior

The **`rescale`** flag controls how strain is applied:

| Value | Effect |
|---|---|
| `y` | Atom coordinates are rescaled with the cell |
| `n` | Only cell parameters change; atoms remain fixed |

Using `rescale = n` effectively concentrates strain at the **periodic boundaries**, which can be useful for fracture or interface studies.

---

## Output behavior

- ReaxFF writes deformation progress to **`fort.77`**
- Each MD step records the updated target and applied cell parameters
- Useful for post-processing strain–stress relationships

---

## Typical use cases

- Crack propagation under controlled strain
- Mechanical testing of crystals and amorphous solids
- Density relaxation of glasses
- Anisotropic lattice deformation studies

---

## Summary

- `vregime.in` enables **dynamic cell manipulation**
- Closely parallels `tregime.in`, but for volume and lattice parameters
- Supports multi-parameter, staged deformation
- Essential for mechanical and strain-driven ReaxFF simulations
