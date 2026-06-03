# ReaxKit

**ReaxKit** is a modular, extensible Python toolkit for **pre‑processing, post‑processing, and analysis of ReaxFF** molecular dynamics simulations.
It provides a clean separation between file parsing, analysis routines, and reproducible workflows, with **Python APIs**, a **CLI interface**, and a **Dash-based GUI**.

ReaxKit is designed for researchers who want a transparent, scriptable bridge between raw ReaxFF data and quantitative, publication‑ready results.

---

## Key Capabilities

### Engine
- Shared engine abstractions for consistent parsing and generating input/output files across 3 different ReaxFF simulation engines, namely `ams`, `lammps`, `reaxff`
- Example IO and generator modules under the `reaxff` engine:
  - ReaxFF input and output files such as `xmolout`, `fort.7`, `fort.13`, `molfra`, and more
  - Input generators for `control`, `geo`, `eregime`, `tregime`, and related files

### Analysis
- Engine-agnostic analyzers built on ReaxKit's domain data models
- Analyzer tasks organized around separated request, result, and task objects
- Analysis components separated from engine IO so the same computations can be reused across scripts, workflows, web UI, and presentation modules

### Workflows
- Reproducible workflows organized by data handling, file tools, meta orchestration, presentation, and study design
- Automation paths for common ReaxFF pre-processing, analysis, post-processing, and presentation tasks

### Web UI
- Dash-based interface components for interactive ReaxFF data inspection
- Backend, UI, and presentation layers for browser-driven workflows

### Presentation
- Publication-ready plotting utilities:
  - 2D plots, dual-axis plots, tornado plots, 3D scatter, and heatmaps
- Video generators

### Utilities
- Shared data, media, and numerical utilities
- Common infrastructure used by analysis, workflows, web UI, and presentation modules

See the full documentation (API reference, tutorials, examples) on [ReaxKit Site](https://ali-m-dinani.github.io/reaxkit/).

---

## Project Layout

```
src/reaxkit/
├── analysis/        # Engine-agnostic analysis tasks
├── cli/             # Command-line entry points
├── core/            # Registries and shared core infrastructure
├── data/            # Packaged reference data and resources
├── domain/          # Central data models, requests, and results
├── engine/          # IO handlers and input generators
├── help/            # Introspection and help system
├── presentation/    # Plotting, active-site views, and media presentation
├── utils/           # Shared data, media, and numerical utilities
├── webui/           # Dash/web interface backend, UI, and presentation layers
└── workflows/       # Workflow orchestration and automation
```

---

##  Testing

Run unit tests with:

```bash
pytest -s tests/
```

to test the package and get the timing for their execution.

---

## Citation

If you use ReaxKit in your work, please cite:

```text
Dinani, A. M., van Duin, A., Shin, Y. K., & Sepehrinezhad, A. (2025).
ReaxKit: A modular Python toolkit for ReaxFF simulation analysis.
Zenodo. https://doi.org/10.5281/zenodo.18485384

Source code: https://github.com/ali-m-dinani/reaxkit
```

---

## Future Directions

* Implement machine learning-based surrogate models for rapid property prediction and active-site identification.
* Develop a plugin system for user-contributed analysis routines and visualization components.
* Integrate with Jupyter notebooks for interactive analysis and visualization.

If you have any feature request, you can submit it through the ReaxKit's GitHub page, or directly sending an email to `Dinani@psu.edu`.

---
## Additional resources:

- [AUTHORS.md](AUTHORS.md) —  Full credits and acknowledgments.
- [LICENSE](LICENSE) — Full license terms under the **MIT License**