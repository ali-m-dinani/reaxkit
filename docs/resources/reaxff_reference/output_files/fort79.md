# fort.79 File

The **fort.79** file reports the **parabolic extrapolation procedure**
used by ReaxFF to optimize force-field parameters during a force-field
optimization run.

It provides the *mathematical justification* for the optimized parameter
values that appear in `fort.13` and are subsequently written back into
the force field.

---

## Purpose

- Fit a parabola through three trial parameter values and their total errors
- Decide whether **interpolation**, **extrapolation**, or **no update** is appropriate
- Select a new parameter value subject to physical and numerical limits

This file is generated **once per parameter** listed in the `params` file.

---

## Optimization Logic

For each parameter, ReaxFF evaluates the total force-field error for:

1. Lower-bound parameter value  
2. Upper-bound parameter value  
3. Original parameter value  

These three points are used to fit a parabola:

\[
E(x) = a x^2 + b x + c
\]

Decision logic:

- **a > 0** → valid minimum exists  
  - Interpolate or extrapolate to the minimum
  - Respect `parext` extrapolation limits
- **a < 0** → hill-shaped parabola  
  - Extrapolation is unsafe
  - Use the best of the three tested values

---

## Example: `fort.79` Output

### Parameter `1 1 1`

```text
Values used for parameter 1 1 1
0.4500000000E+02 0.5500000000E+02 0.5000000000E+02

Differences found
0.9152086304E+02 0.9158925608E+02 0.9164489675E+02

Parabol: a= -0.3593487269E-02 b= 0.3661880309E+00 c= 0.8231921337E+02

Minimum of the parabol 0.5095162492E+02
Difference belonging to minimum of parabol 0.9164814892E+02

New parameter value 0.4500000000E+02
Difference belonging to new parameter value 0.9152086100E+02
```

**Interpretation**

- `a < 0` → hill-shaped parabola
- Extrapolation rejected
- Best existing value retained

---

### Parameter `1 2 1`

```text
Values used for parameter 1 2 1
0.9742293000E+01 0.9939107000E+01 0.9840700000E+01

Differences found
0.9160380360E+02 0.9143296226E+02 0.9151752206E+02

Parabol: a= 0.8889634892E-01 b= -0.2617639062E+01 c= 0.1086682558E+03

Minimum of the parabol 0.1472298410E+02
Difference belonging to minimum of parabol 0.8939852465E+02

New parameter value 0.9988802535E+01
Difference belonging to new parameter value 0.9139091180E+02
```

**Interpretation**

- `a > 0` → valid minimum exists
- Optimal value lies **outside** allowed extrapolation range
- Parameter clipped to nearest allowed value

---

## Relation to Other Files

| File | Role |
|-----|-----|
| `params` | Defines which parameters are optimized |
| `fort.13` | Total error for each trial |
| `fort.79` | Parabolic fit and optimization decision |
| `fort.99` | Breakdown of error contributions |

---

## Practical Notes

- `fort.79` explains *why* a parameter was updated (or not)
- Large extrapolations often signal:
  - Poorly conditioned training sets
  - Overly large search intervals
  - Strong parameter correlations
- This one-parameter approach is robust and reversible, making it suitable
  for long optimization campaigns

---

## ReaxKit Context

In ReaxKit, `fort.79` is especially useful for:

- Debugging force-field optimization behavior
- Auditing parameter updates
- Visualizing optimization trajectories per parameter
