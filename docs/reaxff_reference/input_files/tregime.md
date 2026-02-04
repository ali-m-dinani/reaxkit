# `tregime.in` file — Temperature regimes for MD simulations

The **`tregime.in`** file defines **temperature regimes** for **NVT** or **NPT** molecular dynamics (MD) simulations in ReaxFF.

It allows the system to be divided into **multiple temperature zones**, each of which can follow independent temperature schedules over different stages of the simulation. This enables controlled heating, cooling, and annealing protocols.

---

## Purpose of `tregime.in`

When present, `tregime.in` allows you to:

- Apply different temperatures to different atom groups
- Heat or cool selected regions at controlled rates
- Perform staged annealing simulations
- Monitor zone-resolved temperature evolution

When `tregime.in` is used, ReaxFF generates an additional output file:

- **`fort.75`** — temperature information for each defined zone

---

## File format

Each non-comment line in `tregime.in` defines **one temperature stage**.

Comment lines start with `#`.

### Column definition

```
# Start  #Zones  At1  At2  Tset1  Tdamp1  dT1/dIt   At3  At4  Tset2  Tdamp2  dT2/dIt
```

Where:

- **Start** — MD iteration at which this stage begins  
- **#Zones** — number of temperature zones  
- **At1–At2** — atom index range for zone 1  
- **Tset1** — target temperature for zone 1 (K)  
- **Tdamp1** — temperature damping constant for zone 1  
- **dT1/dIt** — temperature change per MD iteration for zone 1 (K/iteration)  
- **At3–At4**, **Tset2**, **Tdamp2**, **dT2/dIt** — same quantities for zone 2  

Additional zones follow the same pattern.

---

## Example 2.12: `tregime.in` input file

```text
# Start  #Zones  At1  At2  Tset1  Tdamp1  dT1/dIt  At3  At4  Tset2  Tdamp2  dT2/dIt
0        2       1    20   200.0  50.0    0.05     21   40   10.0   1.0     0.00
10000    2       1    20   700.0  50.0    0.00     21   40   10.0   1.0     0.00
20000    2       1    20   700.0  50.0   -0.10     21   40   10.0   1.0     0.00
```

---

## Interpretation of the example

This example divides the system into **two temperature zones**:

- **Zone 1:** atoms **1–20**
- **Zone 2:** atoms **21–40**

### Stage 1 (iteration 0–10,000)
- Zone 1 is heated from **200 K to 700 K**
- Heating rate: **+0.05 K/iteration**
- Zone 2 is held constant at **10 K**

### Stage 2 (iteration 10,000–20,000)
- Zone 1 remains at **700 K**
- Zone 2 remains at **10 K**

### Stage 3 (iteration 20,000–end)
- Zone 1 is cooled at **−0.10 K/iteration**
- Zone 2 remains at **10 K**

This setup performs a **localized annealing protocol**, where only part of the system is thermally driven.

---

## General notes

- There is **no fixed limit** on:
  - Number of temperature zones
  - Number of temperature stages
- Temperature zones may overlap only if physically meaningful
- `tregime.in` is compatible with both **NVT** and **NPT** simulations
- The temperature evolution of each zone is written to **`fort.75`**

---

## Typical use cases

- Surface annealing while keeping bulk atoms cold
- Thermal activation of reaction sites
- Localized heating for diffusion studies
- Multi-step heating–cooling schedules

`tregime.in` provides fine-grained thermal control and is a powerful tool for advanced ReaxFF MD workflows.
