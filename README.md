# ReaxKit

**ReaxKit** is a modular, extensible Python toolkit for **pre‑processing, post‑processing, and analysis of ReaxFF** molecular dynamics simulations.
It provides a clean separation between file parsing, analysis routines, and reproducible workflows, with both **Python APIs** and a **CLI interface**.

ReaxKit is designed for researchers who want a transparent, scriptable bridge between raw ReaxFF files and quantitative, publication‑ready results.

---

## Key Capabilities

### File IO (Parsing & Generation)
- Robust handlers for ReaxFF input and output files:
  - `xmolout`, `fort.7`, `fort.13`, `molfra`, and more
- Input file generators:
  - `control`, `geo`, `eregime`, `tregime`, and more
- Unified handler interface for consistent data access

### Analysis
- Per‑file analyzers (one analyzer per ReaxFF file type)
- Composed analyzers that combine multiple data sources:
  - Coordination numbers
  - Connectivity graphs
  - Electrostatics and dipoles
  - Radial distribution functions (RDF)
- Numerical utilities for smoothing, extrema detection, and signal processing

### Workflows
- Reproducible, CLI‑driven workflows for:
  - Single‑file analysis
  - Multi‑file composed analysis
  - Plotting and media generation
- Designed to automate common ReaxFF post‑processing tasks

### Visualization & Media
- Publication‑ready plotting utilities (2D, dual‑axis, tornado plots, 3D scatter, heatmaps)
- Trajectory and plot video generation

See the full documentation (API reference, tutorials, examples) on [ReaxKit Site](https://ali-m-dinani.github.io/reaxkit/).

---

## Project Layout

```
src/reaxkit/
├── analysis/        # Analysis routines (per-file and composed)
├── io/              # File handlers and generators
├── utils/           # Shared utilities (aliases, units, constants, numerics)
├── workflows/       # CLI and automation workflows
├── help/            # Introspection and help system
└── cli.py           # Command-line entry point
```

Additional resources:

- [Installation notes](docs/installation.md) — Full installation instructions (requires **Python ≥ 3.9**).
- [Quickstart](docs/quickstart.md) — Get up and running with core ReaxKit workflows in minutes.
- [Tutorials notes](docs/tutorials/index.md) and [source files](https://ali-m-dinani.github.io/reaxkit/tutorials/) — Step-by-step guides for common ReaxKit workflows and use cases.
- [Examples](docs/examples/README.md) and [source files](https://ali-m-dinani.github.io/reaxkit/examples/) — Minimal, runnable Python examples using public APIs.
- [ReaxFF Reference](docs/reaxff_reference/index.md) — Reference documentation for ReaxFF input and output files.
- [Contributing](docs/contributing.md) — Guidelines for contributing to ReaxKit.
- [File Templates](docs/file_templates/index.md) and [Docstring Conventions](docs/file_templates/docstrings.md) — Development guidelines and code templates.

---

##  Testing

Run unit tests with:

```bash
pytest -s tests/
```

---

##  Acknowledgements

See [AUTHORS.md](AUTHORS.md) for full credits and acknowledgments.

---

## License

Licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full terms.

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

* Add support for other analyzers including autocorrelation functions, thermodynamic properties calculations, etc.
* Develop GUI dashboard for interactive ReaxFF data inspection
* Implement ML-based trend prediction for simulation outputs

If you have any feature request, you can submit it through the ReaxKit's GitHub page, or directly sending an email to `Dinani@psu.edu`.