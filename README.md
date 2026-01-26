#  ReaxKit

**ReaxKit** is a modular Python toolkit for **pre- and post-processing ReaxFF molecular dynamics simulation files**.  
It provides clean, extensible interfaces for reading and generating AMS/ReaxFF input and output files (e.g., `xmolout`, `control`), processing simulation data, and producing quantitative analyses and graphical visualizations.

Reaxkit employs an object-oriented internal design for parsing and analysis components, while exposing a pipeline-based (pipes-and-filters) workflow architecture for composing data processing and analysis steps.

##  Features

- **Flexible File Handlers**
  - Parse ReaxFF input/outputs including:
    - `xmolout` → atomic trajectories
    - `control` → control settings for the MD simulation
  - Generate inout files such as "eregime.in" using `eregime_generator.py`

- **Analysis Tools**
  - Compute and visualize:
    - Bond orders  
    - Dipole moments  
    - Mean square displacement (MSD)  
    - Coordination numbers, defect tracking, etc.
  - Integrates with **NumPy**, **pandas**, **SciPy**, and **Matplotlib** for efficient data manipulation.

- **Workflows**
  - Chain multiple handlers and analyzers into reproducible pipelines
  - Automate extraction and plotting across multiple ReaxFF runs

- **Extensible Architecture**
  - Designed around classes and interfaces:
    - `Handler` → file parsing  
    - `Analyzer` → feature computation  
    - `Workflow` → automation layer
  - Easily extendable for new file types or analysis routines

---

##  Project Structure

```

ReaxKit/
├── reaxkit/
│   ├── analysis/
│   ├── io/
│   ├── utils/
│   ├── workflows/
│   ├── __init__.py
│   └── cli.py
│
├── examples/
│   └── sample simulations with their analyzed files.
│
├── tests/
│   └── sample files with Jupyter notebooks for analysis.
│
├── LICENSE
├── AUTHORS.md
├── README.md
└── pyproject.toml


````





---

##  License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.



