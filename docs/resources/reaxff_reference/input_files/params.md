# `params` file — Force field parameter optimization control

The **`params`** file specifies **which force field parameters** are included in a ReaxFF **force field optimization** and defines **how each parameter is searched** during optimization.

It tells ReaxFF:
- *Which* parameter to optimize
- *How far* to perturb it during the search
- *What bounds* constrain the optimization

---

## Purpose of the `params` file

During force field optimization, ReaxFF does **not** vary all parameters automatically. Instead, it iterates through the parameters listed in `params`, optimizing them **one at a time** using a constrained parabolic extrapolation.

For each parameter:
1. ReaxFF perturbs the parameter value
2. Recalculates the cost function (from `trainset.in`)
3. Fits a parabola to determine an improved value
4. Applies bounds and extrapolation limits

Results of each optimization step are written to **`fort.79`**.

---

## File handling

- The `params` file is copied by the `exe` script to **`fort.21`**
- ReaxFF reads optimization instructions exclusively from `fort.21`
- The filename `params` itself is not read directly by ReaxFF

---

## File format

Each non-comment line defines **one optimizable force field parameter**.

**Line format**
```
<section> <type> <parameter> <step> <max> <min>
```

Using fixed-width formatting:
```
3i3, 3f8.4
```

---

## Field definitions

| Field | Meaning |
|---|---|
| `section` | Force field section index |
| `type` | Atom/bond/angle/torsion type index |
| `parameter` | Parameter index within that type |
| `step` | Relative perturbation factor |
| `max` | Maximum allowed value |
| `min` | Minimum allowed value |

---

## Force field section indices

The `section` integer maps to the corresponding section in the `ffield` file:

| Index | Section |
|---|---|
| `1` | General parameters |
| `2` | Atom parameters |
| `3` | Bond parameters |
| `4` | Off-diagonal parameters |
| `5` | Valence angle parameters |
| `6` | Torsion angle parameters |
| `7` | Hydrogen bond parameters |

---

## Example 2.16: `params` file

```text
1 1 1 0.100 75.000 25.000   ! 1st general parameter
1 2 1 0.010 10.000  9.000   ! 2nd general parameter
2 1 1 0.001  1.600  1.350   ! section 2, type 1, parameter 1
2 2 1 0.001  0.700  0.600   ! section 2, type 2, parameter 1
2 2 5 0.010  0.020  0.030   ! section 2, type 2, parameter 5
6 3 4 0.050 -3.000 -5.000   ! section 6, type 3, parameter 4
```

---

## Interpretation of an example line

Consider the final line:

```text
6 3 4 0.050 -3.000 -5.000
```

This instructs ReaxFF to optimize:
- **Section**: 6 → torsion angle parameters
- **Type**: 3 → specific torsion type
- **Parameter**: 4 → fourth torsion parameter
- **Search step**: ±5% relative perturbation
- **Bounds**: constrained between −5.000 and −3.000

If the starting value in `ffield` is −4.7435, ReaxFF will:
- Increase and decrease the value by 5%
- Evaluate the cost function
- Perform a parabolic extrapolation
- Choose an improved value within the specified bounds

---

## Optimization workflow

- ReaxFF processes parameters **sequentially**, top to bottom
- Each parameter is optimized independently
- After the final line is processed, the optimization run terminates
- Detailed extrapolation results are written to **`fort.79`**

---

## Interaction with `control` file

Two `control` keywords affect parameter optimization behavior:

| Keyword | Effect |
|---|---|
| `parext` | Limits how far extrapolation may extend beyond the search interval |
| `parsca` | Scales the search interval size globally |

Because the `control` file is reread during optimization, these values can be adjusted **mid-run** to influence long optimizations.

---

## Best practices

- Start with **small step sizes** (`step ≈ 0.001–0.05`)
- Use **tight bounds** to prevent unphysical parameters
- Optimize **most sensitive parameters first**
- Monitor `fort.79` to verify convergence behavior

---

## Summary

- `params` defines **which force field parameters are optimized**
- Each line controls one parameter’s search behavior
- Optimization proceeds sequentially and deterministically
- Central to reproducible and controlled ReaxFF force field fitting
