# `control` file — Run control parameters for ReaxFF

The **`control`** file defines all **run-control parameters** used by ReaxFF, including molecular dynamics (MD), molecular mechanics (MM), and force field optimization settings.

Each non-comment line contains:
```
<number> <keyword> <optional description>
```

Lines starting with `#` are treated as comments and ignored.

---

## Key properties of the `control` file

- **Format-free**: keywords can appear in any order  
- **Optional keywords**: omitted keywords take default values  
- **Live reloading**: ReaxFF periodically re-reads the `control` file during MD or force-field optimization runs, allowing parameters to be modified *while a simulation is running*  
- **Logical grouping only**: the division into *general*, *MD*, *MM*, and *optimization* parameters is for documentation convenience and has no deeper meaning  

---

## Example 2.9: `control` file for an NVT MD simulation

```text
# General parameters
     0 imetho   0: Normal MD-run 1: Energy minimisation 2: MD-energy minimisation
     1 igeofo   0: xyz-input 1: Biograf input 2: xmol-input geometry
80.000 axis1 a (for non-periodical systems)
80.000 axis2 b (for non-periodical systems)
80.000 axis3 c (for non-periodical systems)
0.0010 cutof2 BO-cutoff for valency angles and torsion angles
 0.300 cutof3 BO-cutoff for bond order for graphs
     3 icharg   Charges. 1:EEM 2:- 3: Shielded EEM 4: Full system EEM 5: Fixed
     1 ichaen   Charges. 1:include charge energy 0: Do not include charge energy
    25 irecon  Frequency of reading control-file

# MD parameters
     1 imdmet   MD-method. 1:NVT 3:NVE 4:NPT
 0.250 tstep MD-time step (fs)
050.00 mdtemp 1st MD-temperature
     2 itdmet   Temperature control method
  25.0 tdamp1 Temperature damping constant (fs)
 01000 nmdit Number of MD-iterations
   005 iout1  Output to fort.71 and fort.73
  0050 iout2 Save coordinates
 00025 irten Remove rotational/translational energy
 00025 itrafr Frequency of trarot calls
     1 iout3    Suppress moldyn.xxxx files
     1 iravel   Random initial velocities
000500 iout6 Save velocity file
 02.50 range Back-translation range (Å)
```

**Figure 1** illustrates the logical grouping of parameters inside the `control` file.

---

## Table 1: General keywords

| Keyword | Default | Description |
|---|---|---|
| imetho | 0 | 0: MD; 1: MM; 2: MD energy minimization |
| igeofo | 0 | 0: `.geo`/z-matrix; 1: `.bgf`; 2: `.xyz` |
| axis1 | 200.0 | a-cell size for non-periodic systems |
| axis2 | 200.0 | b-cell size for non-periodic systems |
| axis3 | 200.0 | c-cell size for non-periodic systems |
| cutof2 | 0.001 | BO cutoff for angles/torsions |
| cutof3 | 0.300 | BO cutoff for graphs and `fort.7` |
| icharg | 3 | Charge model (EEM, Shielded EEM, ACKS2, etc.) |
| ichaen | 1 | Include charge energy |
| iappen | 0 | Append or overwrite `fort.7` / `fort.8` |
| isurpr | 0 | Output verbosity level |
| icheck | 0 | Single-point / derivative checks |
| idebug | 0 | Debug output to `fort.65` |
| ixmolo | 0 | Extra data written to `xmolout` |
| itrout | 0 | Generate unfolded trajectory |
| iexx | 1 | Extra periodic images in x |
| iexy | 1 | Extra periodic images in y |
| iexz | 1 | Extra periodic images in z |
| cutmo1 | 0.0 | Extra molecule output cutoff |
| cutmo2 | 0.0 | Extra molecule output cutoff |
| ignore | 0 | Ignore bonds involving atom type |
| ireflx | 0 | Reflective x-boundary |
| irefly | 0 | Reflective y-boundary |
| ireflz | 0 | Reflective z-boundary |

---

## Table 2: MD-related keywords

| Keyword | Default | Description |
|---|---|---|
| imdmet | 3 | 1: NVT; 3: NVE; 4: NPT |
| tstep | 0.25 | Time step (fs) |
| mdtemp | 298.0 | Target temperature |
| itdmet | 2 | Temperature control scope |
| tdamp1 | 2.5 | Temperature damping (fs) |
| mdpres | 0.0 | Pressure (GPa) |
| pdamp1 | 500.0 | Pressure damping (fs) |
| inpt | 0 | Fixed cell directions in NPT |
| nmdit | 1000 | MD iterations |
| ichupd | 1 | Charge update frequency |
| iout1 | 5 | Output to `fort.71` / `fort.73` |
| iout2 | 50 | Output to `xmolout` |
| ivels | 0 | Velocity initialization |
| iout3 | 0 | Write moldyn trajectories |
| iravel | 0 | Random velocities |
| endmd | 1.0 | MD minimization RMSG |
| iout6 | 1000 | Restart file frequency |
| irten | 25 | Remove rotation/translation |
| npreit | 0 | Previous MD iterations |
| range | 2.50 | Back-translation range (Å) |
| irecon | 25 | Control file reread frequency |
| iremov | 0 | Molecule removal |
| vramin | 0.1 | Upper mass cutoff |
| vrami2 | 0.01 | Lower mass cutoff |

---

## Table 3: MM-related keywords

| Keyword | Default | Description |
|---|---|---|
| endmm | 1.0 | MM RMSG termination |
| imaxmo | 50 | Step control / CG |
| imaxit | 50 | Max MM iterations |
| iout4 | 50 | MM structure output |
| iout5 | 0 | Remove `fort.57` / `fort.58` |
| icelop | 0 | Cell optimization |
| celopt | 1.0005 | Cell optimization step |
| icelo2 | 0 | Cell optimization mode |

---

## Table 4: Force field optimization keywords

| Keyword | Default | Description |
|---|---|---|
| parsca | 1.0 | Parameter scaling |
| parext | 0.001 | Parameter extrapolation |
| igeopt | 1 | Geometry update control |
| iincop | 0 | Heat increment optimization |
| accerr | 2.50 | Accepted error increase |
