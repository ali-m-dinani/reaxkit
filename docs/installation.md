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

This installs the core package and core dependencies only.

---

## Install with extras

Use extras based on your workflow:

```bash
pip install "reaxkit[plot]"
pip install "reaxkit[trajectory]"
pip install "reaxkit[webui]"
pip install "reaxkit[all]"
```

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

```bash
git clone https://github.com/ali-m-dinani/reaxkit.git
cd reaxkit
pip install .
```

For editable development install:

```bash
pip install -e .
```

or with extras:

```bash
pip install -e ".[plot,trajectory]"
```

---

## Verify installation

```bash
reaxkit --help
```

If the help message appears, installation is successful.

---

## Notes

- If `gen-video` fails, ensure `ffmpeg` is installed and available in `PATH`.
- Some optional backends may have platform-specific install constraints.
