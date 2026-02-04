# fort.13 File

The **fort.13** file reports the *total force-field error* during a ReaxFF force‑field
optimization. The error is computed from the cost function defined in
`trainset.in` and reflects how well the current force‑field parameters
reproduce the reference data.

This file is only generated during **force field optimization runs**.

---

## Purpose

- Track how the **total error** changes as individual force‑field parameters
  are varied.
- Provide input for the **parabolic extrapolation** procedure used to determine
  optimized parameter values.
- Act as a high‑level convergence and diagnostics file for parameter fitting.

---

## When Is `fort.13` Written?

`fort.13` is written when ReaxFF is run with:

- `iopt = 1` (force field optimization mode)
- A valid `params` file
- A valid `trainset.in` defining the cost function

Each parameter listed in `params` produces **four entries** in `fort.13`.

---

## Example: `fort.13` Output

```text
91.5209  ! Total FF error using 45.00 for parameter 1 1 1
91.5893  ! Total FF error using 55.00 for parameter 1 1 1
91.6449  ! Total FF error using 50.00 for parameter 1 1 1
91.5209  ! Total FF error using optimized value for parameter 1 1 1 (see fort.79)

91.6038  ! Total FF error using 9.7423 for parameter 1 2 1
91.4330  ! Total FF error using 9.9391 for parameter 1 2 1
91.5175  ! Total FF error using 9.8407 for parameter 1 2 1
91.3896  ! Total FF error using optimized value for parameter 1 2 1 (see fort.79)

............ ! Next parameter listed in params
```

---

## Interpretation

For **each parameter** listed in the `params` file, ReaxFF performs:

1. **Lower-bound test**  
   Parameter value decreased by the search interval

2. **Upper-bound test**  
   Parameter value increased by the search interval

3. **Baseline test**  
   Original force‑field value

4. **Optimized test**  
   Value obtained from parabolic extrapolation (see `fort.79`)

Each of these runs produces one line in `fort.13`.

---

## Relation to Other Files

| File      | Role |
|-----------|------|
| `params`  | Defines which parameters are optimized and their search bounds |
| `trainset.in` | Defines the cost function |
| `fort.13` | Reports total error for each trial |
| `fort.79` | Contains details of the parabolic extrapolation |
| `fort.99` | Breaks down individual error contributions |

---

## Practical Notes

- **Lower values** in `fort.13` indicate a better fit to the training set.
- The optimized value is always the **4th entry** per parameter.
- Sudden increases in error often indicate:
  - Overly large search intervals
  - Conflicting training-set targets
  - Over‑constrained linked parameters (`koppel2`)

---

## ReaxKit Usage

In ReaxKit, `fort.13` is typically used to:

- Track optimization progress
- Detect unstable parameter updates
- Summarize force‑field quality during fitting workflows
