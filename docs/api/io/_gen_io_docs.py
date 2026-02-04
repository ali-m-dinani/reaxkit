# _gen_io_docs.py
"""
Generate mkdocstrings IO reference pages for ReaxKit.

This script walks:
  reaxkit/src/reaxkit/io/

and writes Markdown pages into:
  reaxkit/docs/api/io/

For each Python module, it creates a Markdown file with:

    # <Title> IO

    ::: reaxkit.io.<subpackages>.<module>

It mirrors folder structure (e.g., io/handlers/control_handler.py ->
docs/api/io/handlers/control_handler.md).

Run
---
From anywhere:
    python reaxkit/docs/api/io/_gen_io_docs.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def _to_title_from_stem(stem: str) -> str:
    """Convert a filename stem like 'control_handler' into 'Control Handler'."""
    parts = [p for p in stem.strip().split("_") if p]
    return " ".join(p[:1].upper() + p[1:] for p in parts) if parts else stem


def _iter_py_modules(io_dir: Path, *, include_private: bool) -> Iterable[Path]:
    """Yield .py files under io_dir respecting skip rules."""
    for py in sorted(io_dir.rglob("*.py")):
        if py.name == "__init__.py":
            continue
        if not include_private and py.name.startswith("_"):
            continue
        yield py


def _module_qualname_from_io(io_dir: Path, py_file: Path) -> str:
    """
    Convert:
      <...>/reaxkit/src/reaxkit/io/handlers/control_handler.py
    into:
      reaxkit.io.handlers.control_handler
    """
    rel = py_file.relative_to(io_dir)         # e.g. handlers/control_handler.py
    parts = ["reaxkit", "io", *rel.parts]
    parts[-1] = py_file.stem                  # drop .py
    return ".".join(parts)


def _docs_md_path(analysis_dir: Path, py_file: Path, docs_root: Path) -> Path:
    """
    Mirror folder structure under docs_root:
      analysis_dir/foo/bar.py -> docs_root/foo/bar_doc.md
    """
    rel = py_file.relative_to(analysis_dir)          # foo/bar.py
    rel = rel.with_name(f"{rel.stem}_doc.md")        # foo/bar_doc.md
    return docs_root / rel


def _write_md(md_path: Path, *, title: str, module: str) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(f"# {title} \n\n::: {module}\n", encoding="utf-8")


def _auto_paths(script_path: Path) -> tuple[Path, Path]:
    """
    Auto-detect:
      repo_root = <...>/reaxkit
      src_root  = repo_root/src
      docs_root = repo_root/docs/api/io
    based on current script location: repo_root/docs/api/io/_gen_io_docs.py
    """
    # .../reaxkit/docs/api/io/_gen_io_docs.py
    docs_io_dir = script_path.resolve().parent          # .../docs/api/io
    repo_root = docs_io_dir.parents[2]                  # .../reaxkit (io -> api -> docs -> reaxkit)
    src_root = repo_root / "src"
    docs_root = repo_root / "docs" / "api" / "io"
    return src_root, docs_root


def generate_io_docs(
    *,
    src_root: Path,
    docs_root: Path,
    include_private: bool = False,
) -> list[Path]:
    """
    Generate Markdown reference pages for modules in reaxkit.io.

    Parameters
    ----------
    src_root : pathlib.Path
        Path to the `src/` directory containing the `reaxkit/` package.
    docs_root : pathlib.Path
        Output directory for docs pages (e.g., `reaxkit/docs/api/io/`).
    include_private : bool, optional
        If True, include modules whose filenames start with '_' (still skips __init__.py).

    Returns
    -------
    list[pathlib.Path]
        Paths of generated Markdown files.
    """
    io_dir = src_root / "reaxkit" / "io"
    if not io_dir.exists():
        raise FileNotFoundError(f"Could not find io directory: {io_dir}")

    written: list[Path] = []
    for py in _iter_py_modules(io_dir, include_private=include_private):
        module = _module_qualname_from_io(io_dir, py)
        title = _to_title_from_stem(py.stem)
        out_md = _docs_md_path(io_dir, py, docs_root)
        _write_md(out_md, title=title, module=module)
        written.append(out_md)

    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate mkdocstrings reference pages for reaxkit.io modules."
    )
    parser.add_argument(
        "--src-root",
        default=None,
        help="Path to the src/ directory (auto-detected if omitted).",
    )
    parser.add_argument(
        "--docs-root",
        default=None,
        help="Output docs directory (auto-detected if omitted).",
    )
    parser.add_argument(
        "--include-private",
        action="store_true",
        help="Include modules whose filenames start with '_' (still skips __init__.py).",
    )
    args = parser.parse_args()

    script_path = Path(__file__)
    auto_src_root, auto_docs_root = _auto_paths(script_path)

    src_root = Path(args.src_root).resolve() if args.src_root else auto_src_root
    docs_root = Path(args.docs_root).resolve() if args.docs_root else auto_docs_root

    written = generate_io_docs(
        src_root=src_root,
        docs_root=docs_root,
        include_private=bool(args.include_private),
    )

    print(f"[Done] Generated {len(written)} io doc pages under: {docs_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
