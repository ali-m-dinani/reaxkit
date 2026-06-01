# `eregime.in` file — Electric field control during MD

The **`eregime.in`** file is an **optional input** that allows ReaxFF to impose **external electric fields** during a simulation.

The applied electric field is **fully coupled to the EEM charge model**, meaning that the field **polarizes the system self-consistently**. This enables simulations of field-driven processes such as polarization, dielectric response, and electrochemical effects.

---

## Purpose of `eregime.in`

With `eregime.in`, users can:

- Turn electric fields **on or off** during a simulation
- Apply fields along specific directions (`x`, `y`, `z`)
- Define **time-dependent field schedules**
- Apply **multiple fields simultaneously** (e.g., `x` and `y`)

When `eregime.in` is present, ReaxFF generates an additional output file:

- **`fort.78`** — electric field strength in each direction and the associated field energy term

---

## Important physical limitations

⚠️ **Periodic boundary caution**  
Electric fields do **not function correctly** if molecules cross a periodic boundary **along the field direction**. This leads to **energy discontinuities**.

**Best practice**
- Use a **large vacuum layer** along the field direction
- Ensure molecules remain within the same periodic image
- Ideal for slab, surface, or capacitor-like geometries

---

## General properties

- **Format-free** input
- **Stage-based** field definition
- Supports **multiple simultaneous fields**
- Compatible with EEM-based charge models

---

## File format

Each non-comment line defines **one electric-field regime stage**.

Comment lines begin with `#`.

### Column definition (conceptual)

```
start  #V  direction  magnitude(V/Å)  [direction  magnitude(V/Å)] ...
```

Where:

- **start** — MD iteration at which this field stage begins  
- **#V** — number of electric-field components defined  
- **direction** — field direction (`x`, `y`, or `z`)  
- **magnitude** — field strength in **V/Å**  

Additional direction–magnitude pairs may follow on the same line.

---

## Example 2.18: `eregime.in` input file

```text
# Electric field regimes
# start  #V  direction  Magnitude (V/Angstrom)

0000  1  x  0.010000
1000  1  x -0.010000
2000  1  y  0.010000
3000  1  y -0.010000
4000  2  x -0.010000  y -0.0100
5000  2  x  0.010000  y  0.0100
```

---

## Interpretation of the example

### Stage 1 (iteration 0 →)
- Apply an electric field in the **+x direction**
- Magnitude: **0.01 V/Å**

### Stage 2 (iteration 1000 →)
- Reverse the **x-field** direction

### Stage 3 (iteration 2000 →)
- Apply a field in the **+y direction**

### Stage 4 (iteration 3000 →)
- Reverse the **y-field** direction

### Stage 5 (iteration 4000 →)
- Apply simultaneous fields in **−x** and **−y** directions

### Stage 6 (iteration 5000 →)
- Reverse both fields to **+x** and **+y**

---

## Output behavior

- ReaxFF writes electric-field data to **`fort.78`**
- Output includes:
  - Field components in `x`, `y`, and `z`
  - Total electric-field energy contribution
- Useful for post-processing polarization and dielectric response

---

## Typical use cases

- Polarization and dielectric response studies
- Field-driven surface chemistry
- Electrochemical interface simulations
- Ferroelectric and piezoelectric materials modeling

---

## Summary

- `eregime.in` enables **time-dependent electric fields**
- Fully coupled to **EEM charge polarization**
- Supports multi-directional fields
- Requires care with periodic boundaries
- Essential for field-driven ReaxFF simulations
