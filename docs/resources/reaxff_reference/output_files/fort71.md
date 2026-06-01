# `fort.71` file — MD energy, temperature, and pressure report

The **`fort.71`** file contains **time-series thermodynamic information** from a ReaxFF **molecular dynamics (MD) simulation**.

It is the primary file for monitoring:
- Energy conservation
- Temperature control
- Pressure behavior
- Convergence during MD or MD-energy minimization

---

## Purpose of the `fort.71` file

`fort.71` is used to:

- Track **potential, kinetic, and total energy**
- Monitor **MD temperature statistics**
- Inspect **pressure evolution**
- Diagnose **MD stability and equilibration**
- Determine convergence in MD-energy minimization runs

---

## Output frequency

The output frequency is controlled by the `control`-file keyword:

```
iout1
```

A line is written to `fort.71` every `iout1` MD iterations.

In **Example 3.5**, `iout1 = 5`, so output appears every 5 iterations.

---

## File structure

- No header comments
- One line per output step
- Fixed column order with **16 columns**

---

## Example 3.5: `fort.71` output file

```text
Iter. Nmol Epot Ekin Etot T(K) Eaver(b) Eaver(tot) Taver Tmax Pres sdev1 sdev2 Tset Tstep RMSG

5   1 1 -663.7 0.8 -662.9 38.3 -663.8 -663.8 41.3 46.0 0.0 0.1 0.1 50.0 0.5 22.2
10  1 1 -663.9 0.9 -663.0 42.4 -663.8 -663.8 42.2 44.5 0.0 0.0 0.0 50.0 0.5 28.7
15  1 1 -663.8 0.9 -662.9 43.9 -663.8 -663.8 42.5 43.9 0.0 0.0 0.0 50.0 0.5 18.9
20  1 1 -663.9 1.0 -662.9 49.9 -663.9 -663.8 48.6 49.9 0.0 0.0 0.0 50.0 0.5 7.4
25  1 1 -663.7 0.8 -662.9 37.1 -663.7 -663.8 40.5 45.4 0.0 0.1 0.0 50.0 0.5 25.0
30  1 1 -663.8 0.5 -663.3 24.6 -663.8 -663.8 21.5 25.6 0.0 0.1 0.0 50.0 0.5 31.1
35  1 1 -663.8 0.5 -663.3 25.0 -663.8 -663.8 25.4 27.0 0.0 0.0 0.0 50.0 0.5 15.0
40  1 1 -663.7 0.5 -663.2 25.3 -663.7 -663.8 24.3 25.4 0.0 0.0 0.0 50.0 0.5 13.1
45  1 1 -663.7 0.6 -663.2 26.9 -663.7 -663.8 24.9 26.9 0.0 0.0 0.0 50.0 0.5 30.0
```

---

## Column definitions

| Column | Description |
|---|---|
| `Iter` | MD iteration number |
| `Nmol` | Number of molecules (using `cutof2` bond criterion) |
| `Epot` | Total potential energy |
| `Ekin` | Total kinetic energy |
| `Etot` | Total energy (`Epot + Ekin`) |
| `T(K)` | Instantaneous MD temperature |
| `Eaver(b)` | Block-averaged potential energy over last `iout1` steps |
| `Eaver(tot)` | Average potential energy over entire run |
| `Taver` | Average temperature over last `iout1` steps |
| `Tmax` | Maximum temperature in last `iout1` steps |
| `Pres` | MD pressure (MPa, intermolecular contribution only) |
| `sdev1` | Std. deviation of `Epot` over last `iout1` steps |
| `sdev2` | Std. deviation of average `Epot` over entire run |
| `Tset` | Target (set) temperature |
| `Tstep` | MD timestep (fs) |
| `RMSG` | Root-mean-square force |

---

## Notes on pressure

The pressure reported in `fort.71`:
- Is based on **intermolecular interactions only**
- Is **not reliable** for evaluating pressure in condensed phases
- Should be interpreted qualitatively rather than quantitatively

---

## Termination behavior

In **MD-energy minimization runs** (`imetho = 2`):

- ReaxFF terminates when:
  - `RMSG < endmd` (from `control` file), **or**
  - Maximum MD iterations are reached

---

## Practical usage

- Primary file for **MD diagnostics**
- Often plotted as:
  - `Epot vs iteration`
  - `Temperature vs iteration`
- Parsed routinely in automated MD analysis pipelines

---

## Summary

- `fort.71` reports MD thermodynamics and convergence
- Output frequency controlled by `iout1`
- Central to equilibration and stability analysis
- Essential companion to `xmolout` for MD workflows
