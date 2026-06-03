# Installation

This page explains how to install ReaxKit and verify the CLI.

---

## Requirements

- Python >= 3.9
- Linux, macOS, or Windows

Core dependencies are installed automatically with `pip`.

Some features need optional packages or external tools:
- plotting packages (via extras)
- trajectory backends (via extras)
- `ffmpeg` for video workflows

---

## Install from PyPI (base)

```bash
pip install reaxkit
```

This installs the **core package and core dependencies** only.

---

## Install with extras

To install optional dependencies for specific features, follow these commands:

```bash
pip install "reaxkit[plot]"
pip install "reaxkit[trajectory]"
pip install "reaxkit[webui]"
pip install "reaxkit[all]"
```

where the last one installs all optional dependencies. 


Available extras include:
- `plot`
- `webui`
- `trajectory`
- `materials`
- `graph`
- `ml`
- `io`
- `engines`
- `all`

---

## Install from source

If you want to install the latest development version (which is not yet publihsed on PyPI, and hence not installable yet using pip) or contribute, you can clone the repository and install from source:

```bash
git clone https://github.com/ali-m-dinani/reaxkit.git
cd reaxkit
pip install .
```

For editable development install (i.e., changes to source code are reflected without reinstalling), use:

```bash
pip install -e .
```

or with extras:

```bash
pip install -e ".[plot,trajectory]"
```

---

## Verify installation

To make sure ReaxKit is installed correctly, run the following command in your terminal:

```bash
reaxkit help -h
```

If the help message appears, installation is successful.

---

## Notes

- If `gen-video` fails, ensure `ffmpeg` is installed and available in `PATH`.
- Some optional backends may have platform-specific install constraints.
