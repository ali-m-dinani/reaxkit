# ReaxKit Examples

This directory contains **runnable example scripts** that demonstrate how to use
ReaxKit in practice. These examples are intended to be:

- minimal,
- reproducible,
- easy to modify,
- executable without Jupyter notebooks.

They complement the tutorials in [tutorials](../tutorials/index.md) by showing **working code**
rather than detailed explanations.

---

## Directory structure

```
examples/
    data/
        control
        sample_tabular_data.csv
        small_fort.7
        small_xmolout
    
    xmolout_basic_example.py
    xmolout_fort7_example.py
    plotter_meta_example.py
    README.md
```


---

## Data files (`examples/data/`)

The `data/` subfolder contains **small, truncated sample files** used by the
example scripts.

These files are intentionally small so that examples:
- run quickly,
- do not require large downloads,
- are suitable for local testing.

### Included files
- `small_xmolout`  
  Minimal `xmolout` trajectory for geometry-based analysis.

- `small_fort.7`  
  Minimal `fort.7` file containing per-atom and bond-order information.

- `control`  
  Example ReaxFF control file (used for reference or input generation examples).

- `sample_tabular_data.csv`  
  Generic tabular data used by the meta plotting example.

---

## Example scripts

### `xmolout_basic_example.py`

Demonstrates **direct Python usage** of ReaxKit analyzers on an `xmolout` file.

Shows how to:
- load `xmolout` with `XmoloutHandler`,
- extract atom trajectories,
- compute derived quantities (e.g. MSD),
- inspect box and thermodynamic information,
- export results to CSV.

This example mirrors the logic used in CLI workflows (as explained by [01_understanding_quickstart.md](../tutorials/01_understanding_quickstart.md)), but uses analyzers directly.

---

### `xmolout_fort7_example.py`

Demonstrates **multi-file analysis** using both `xmolout` and `fort.7`.

Shows how to:
- load geometry (`xmolout`) and chemical data (`fort.7`),
- extract per-atom properties (charges, sum of bond orders),
- construct connectivity graphs,
- track bond-order time series and bond events,
- compute coordination status using atom types and valences,
- export results for further analysis or plotting.

This example corresponds to the concepts explained in [02_xmolout_fort7_coordination.md](../tutorials/02_xmolout_fort7_coordination.md).

---

### `plotter_meta_example.py`

Demonstrates use of ReaxKit’s **meta plotting utilities** on arbitrary tabular data.

Shows how to:
- read a generic CSV file,
- generate single-curve and multi-curve plots,
- create directed plots,
- generate dual y-axis plots,
- visualize 3D scatter data,
- project 3D data into 2D heatmaps.

This example is file-agnostic and does not rely on ReaxFF-specific semantics. A CLI tutorial for this existes in [04_plotter_meta_workflow](../tutorials/04_plotter_meta_workflow.md).

---

## Notes

* These examples are not tests.
* They are meant for learning, exploration, and validation.
* For conceptual explanations, see [tutorials](../tutorials/index.md).
* For extending ReaxKit, see [templates](../file_templates/index.md).

If you encounter issues running an example, verify that ReaxKit is installed
correctly and that paths to the `data/files` are unchanged.








