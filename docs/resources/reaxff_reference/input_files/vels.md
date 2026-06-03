# `vels` file — MD restart file (positions, velocities, accelerations)

The **`vels`** file stores **atomic positions, velocities, and accelerations** and is used to **restart ReaxFF MD simulations**.

During an MD run, ReaxFF periodically writes restart files based on the [control](control.md) keywords `iout2` and `iout6`. These files can be reused to continue a simulation seamlessly.

---

## Purpose of the `vels` file

- Restart an interrupted MD simulation
- Continue dynamics with the latest positions and velocities
- Override the geometry provided in the `geo` file (positions only)

> **Important:** The `geo` file is still required. ReaxFF checks that `geo` and `vels` contain the **same number of atoms** before restarting.

---

## File handling and naming

- During MD, ReaxFF generates:
  - **`moldyn.vel`** — most recent restart snapshot
  - **`molsav.xxxx`** — periodic restart snapshots
- To restart:
  1. Copy a restart file to **`vels`**
  2. Re-run the `exe` script
- The `exe` script then copies `vels` to **`moldyn.vel`**
- ReaxFF reads `moldyn.vel` and **overwrites it** at intervals set by `iout2`

This naming can be confusing, but the flow is consistent once understood.

---

## What the `vels` file contains

In order, the file includes:

1. **Lattice parameters**
2. **Atom coordinates**
3. **Atom velocities**
4. **Atom accelerations**
5. **Previous atom accelerations**
6. **Instantaneous MD temperature**

Acceleration history is included for completeness but **does not influence** restarted runs if outdated.

---

## Example 2.16: `vels` restart file

The following example shows a restart file generated for an MD simulation of an **ethyl radical**.

```text
Lattice parameters:

80.00000000 80.00000000 80.00000000
90.00000000 90.00000000 90.00000000

7 Atom coordinates (Angstrom):

0.392027787892917E+02 0.400902383055691E+02 0.396569719222545E+02 C
0.408076194536754E+02 0.407017914536647E+02 0.409641189620557E+02 H
0.406156358473423E+02 0.399085739070730E+02 0.402446784332288E+02 C
0.389320157142440E+02 0.409742263417328E+02 0.391261065022365E+02 H
0.384743625215192E+02 0.392966529997087E+02 0.397356676283833E+02 H
0.413546771765171E+02 0.399886339278518E+02 0.394389532581619E+02 H
0.407016447482017E+02 0.389398236842467E+02 0.407560383741949E+02 H

Atom velocities (Angstrom/s):

-0.639976248783438E+12  0.451200007339684E+11 -0.216754312886992E+12
-0.703141792567835E+11  0.421377644135005E+13  0.824472191800807E+13
 0.447898432906652E+12 -0.105077751499844E+13 -0.658290512224369E+11
 0.195067105934229E+13  0.289176812368303E+13 -0.140045800761676E+14
 0.476126532155786E+13 -0.152785617001799E+13  0.169491694508477E+14
 0.456467329116016E+13  0.118352257917078E+14 -0.396100285863348E+12
-0.891965482760371E+13 -0.544080092166964E+13 -0.742912333885548E+13

Atom accelerations (Angstrom/s**2):

-0.810436027147472E+27 -0.981147717871880E+27 -0.313616213868668E+27
 0.116753745428499E+29  0.109570643908876E+29  0.135954922986376E+29
-0.244853315374905E+27 -0.599286401168541E+27 -0.831858387309253E+27
 0.105361267406127E+28  0.134697085751095E+28 -0.115819691864393E+27
 0.208416152961380E+28  0.364332842399750E+28 -0.673228416751219E+27
-0.149559883359616E+28 -0.125042567119739E+28  0.468080992010752E+28
-0.754581549567212E+27  0.411775389213967E+28 -0.385065171515426E+28

Previous atom accelerations:

0.000000000000000E+00 0.000000000000000E+00 0.000000000000000E+00
0.000000000000000E+00 0.000000000000000E+00 0.000000000000000E+00
0.000000000000000E+00 0.000000000000000E+00 0.000000000000000E+00
0.000000000000000E+00 0.000000000000000E+00 0.000000000000000E+00
0.000000000000000E+00 0.000000000000000E+00 0.000000000000000E+00
0.000000000000000E+00 0.000000000000000E+00 0.000000000000000E+00
0.000000000000000E+00 0.000000000000000E+00 0.000000000000000E+00

MD-temperature (K):

0.550040311117348E+02
```

---

## Notes and best practices

- **Geometry precedence:** positions in `vels` override those in `geo`
- **Atom count must match** between `vels` and `geo`
- **Acceleration history** is generally not critical for restarts
- Restart files are essential for:
  - Long simulations
  - Queue-limited HPC jobs
  - Crash recovery

---

## Summary

- `vels` enables **robust MD restarts**
- Generated automatically during MD
- Copied to `moldyn.vel` and refreshed during the run
- Central to production-scale ReaxFF workflows
