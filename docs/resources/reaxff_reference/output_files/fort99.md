# fort.99 File

The **fort.99** file provides a **detailed breakdown of how well ReaxFF
reproduces the reference data** defined in `trainset.in`. It is the most
informative output file for diagnosing *why* a force field performs well
or poorly during optimization.

While `fort.13` reports only the **total error**, `fort.99` shows the
**individual contributions** that make up that total.

---

## Purpose

- Compare ReaxFF-computed values against QM or literature references
- Compute weighted squared errors for each training-set entry
- Provide transparency into the force-field cost function

This file is generated during **force-field optimization runs**.

---

## File Structure

Each line corresponds to **one entry in `trainset.in`** and reports:

| Column | Meaning |
|------|--------|
| Description | Type of data (charge, bond, angle, energy, etc.) |
| ReaxFF value | Value computed by ReaxFF |
| QM/Lit value | Reference value |
| Weight | Weight from `trainset.in` |
| Error | Squared weighted error |
| Total error | Cumulative error up to this line |

---

## Example: `fort.99` Output

```text
FField value   QM/Lit value   Weight   Error   Total error

methane Heat of formation:
-17.8000       -17.8000       2.0000   0.0000  0.0000

chexane Charge atom: 1
-0.1604        -0.1500        0.1000   0.0109  0.0109

Heat of formation:
-29.4900       -29.4900       2.0000   0.0000  0.0109

Bond distance: 1 2
1.5586         1.5400         0.0100   3.4571  3.4679

Bond distance: 1 7
1.1696         1.1000         0.0200   12.1227 15.5906

Bond distance: 1 8
1.1713         1.1000         0.0200   12.7203 28.3109

Valence angle: 1 2 3
110.8117       111.0000       1.0000   0.0354  28.3463

Valence angle: 7 1 8
104.3207       107.0000       1.0000   7.1788  35.5251

chex_cryst a:
11.8448        11.2000        0.4000   2.5987  38.1238

Energy +butbenz/1 -butbenz_a/1
-96.6941       -90.0000       1.5000   19.9158 58.0396

Energy +butbenz/1 -butbenz_b/1
-63.4751       -71.0000       1.5000   25.1663 83.2060

Energy +butbenz/1 -butbenz_c/1
-77.1805       -78.0000       1.5000   0.2985  83.5045

Energy +chex_cryst/16 -chexane/1
-6.2139        -11.8300       2.0000   7.8851  91.3896
```

---

## Error Definition

The force-field error is computed as:

\[
\text{Error}^{\text{ReaxFF}} =
\left( \frac{v^{\text{ReaxFF}} - v^{\text{QC/Lit}}}{\text{weight}} \right)^2
\]

The **sum of all individual errors** (final value in the last column)
is the total error used in:

- `fort.13` (summary error per parameter trial)
- `fort.79` (parabolic extrapolation)

---

## Relation to Other Files

| File | Role |
|-----|-----|
| `trainset.in` | Defines target data and weights |
| `fort.99` | Per-datapoint error breakdown |
| `fort.13` | Total error per parameter trial |
| `fort.79` | Parameter optimization logic |

---

## Practical Notes

- Large contributions in `fort.99` immediately highlight:
  - Poorly fit bonds or angles
  - Overweighted training data
  - Conflicting targets in `trainset.in`
- Energy terms often dominate the total error
- `fort.99` is the best starting point for *manual* force-field debugging

---

## ReaxKit Context

In ReaxKit, `fort.99` is ideal for:

- Visualizing error contributions
- Ranking training-set importance
- Building diagnostic plots for force-field optimization
