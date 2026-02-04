# Installation

This document describes how to install **ReaxKit** and verify that it is working
correctly. All other documentation assumes ReaxKit is already installed.

---

## Requirements

- **Python** ≥ 3.9 (recommended: 3.10 or newer)
- Operating system:
  - Linux, macOS, or Windows
  - Windows users may prefer WSL for HPC / cluster workflows

ReaxKit is a pure-Python package but depends on several scientific libraries
(e.g. NumPy, pandas, SciPy, Matplotlib). These are handled automatically by `pip`.

Some optional features may require external tools:
- **OVITO** (for advanced visualization)
- **ffmpeg** (for video generation)

These are not required for core functionality.

---

## Install from PyPI (recommended)

You can easily install ReaxKit and its dependencies using:

```bash
pip install reaxkit
```

---

## Install from source

Clone the repository and install locally:

```bash
git clone https://github.com/ali-m-dinani/reaxkit.git
cd reaxkit
pip install .
```

This is useful if you want the latest development version.

---

## Developer / editable installation

If you plan to modify ReaxKit (handlers, analyzers, workflows):

```bash
pip install -e .
```

This installs ReaxKit in editable mode, so changes to the source code are
picked up immediately without reinstalling.

---

## Optional dependencies

Some features rely on optional tools:

* Plotting / visualization

  * `matplotlib` (installed by default)

  * `ovito` (optional, install separately if needed)

* Video generation

  * Requires `ffmpeg` available on your system

ReaxKit will function without these, but related workflows may be unavailable.

---

## Verify installation

After installation, run:

```bash
reaxkit --help
```

You should see the ReaxKit CLI help message, which shows that ReaxKit is installed correctly.

If you encounter installation issues, please consult the README or open an issue
on the repository.













