# `fort.57` file — MM energy minimization report

The **`fort.57`** file contains a **detailed report of a molecular mechanics (MM) energy minimization** performed by ReaxFF.

It is written during **MM runs**, typically using a **conjugate-gradient minimizer**, and provides iteration-by-iteration information on energy convergence and force reduction.

---

## Purpose of the `fort.57` file

The `fort.57` output is primarily used to:

- Monitor **energy minimization convergence**
- Track reduction of atomic forces (RMSG)
- Diagnose minimizer behavior and stability
- Verify termination criteria for MM runs

---

## When `fort.57` is generated

- Generated during **MM minimization** runs (`imetho = 1` or `2`)
- Updated every MM iteration
- May be deleted automatically if `control` keyword `iout5 = 1`

---

## File structure

The file consists of:

1. A **structure identifier**
2. A **tabular iteration log**

---

## Column definitions

Each iteration line contains the following columns:

| Column | Description |
|---|---|
| `Iter.` | MM iteration number |
| `Epot` | Potential energy (kcal/mol) |
| `Max.move` | Maximum atomic displacement in this iteration |
| `Factor` | Scaling factor used by the minimizer |
| `RMSG` | Root-mean-square gradient of forces |
| `nfc` | Number of force calculations |

---

## Example 3.3: `fort.57` output file

```text
Ethyl_radical

Iter. Epot Max.move Factor RMSG nfc
0 -664.0048203101 0.000000 0.500000 4.969706 0
1 -664.0170216413 15.795587 0.089117 3.360058 0
2 -664.0394728751 9.339616 0.135649 6.279516 0
3 -664.0765683298 0.001387 1.957694 3.047916 0
4 -664.0956555161 0.000342 3.000000 2.270724 0
5 -664.1009627083 0.001933 0.585262 1.026026 0
6 -664.1025221864 0.000851 1.000000 0.465193 0
```

---

## Interpreting the output

### Key quantities

- **Epot**  
  Potential energy of the system at each iteration. A monotonic decrease usually indicates stable minimization.

- **RMSG**  
  Root-mean-square gradient of the forces. This is the **primary convergence metric**.

### Termination criteria

An MM minimization terminates when **either**:

- `RMSG` drops below the threshold defined by `endmm` in the `control` file, **or**
- The maximum number of iterations (`maxit` in `control`) is reached

In Example 3.3, the run terminates when `RMSG < 0.500`.

---

## Practical notes

- `fort.57` is the MM analogue of `fort.73`/`energylog` for MD
- Useful for diagnosing poor initial geometries
- Sudden spikes in `Max.move` or `RMSG` may indicate unstable force-field parameters
- Often parsed automatically in force-field optimization workflows

---

## Summary

- `fort.57` reports MM minimization progress
- Focus on **Epot** and **RMSG** for convergence assessment
- Termination controlled by `endmm` and `maxit`
- Essential for validating MM geometry relaxations
