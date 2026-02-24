"""Tools for extracting one-line summaries from modules and their functions/classes.

This module is used by the `reaxkit intspec` workflow to:
  - list all .py files under a folder (recursively) with module docstring summaries
  - show a module's docstring and a table of public functions/classes with doc summaries
"""

from __future__ import annotations

import ast
import os
import importlib.util
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple


# ---------------------------- helpers ----------------------------

def _first_line(s: str | None) -> str:
    """
    Return the first non-empty line from a docstring-like text.

    Works on
    --------
    Optional multi-line strings

    Parameters
    ----------
    s : str or None
        Source string.

    Returns
    -------
    str
        First meaningful line, or ``"No description"``.

    Examples
    --------
    >>>
    """
    if not s:
        return "No description"
    for line in s.strip().splitlines():
        if line.strip():
            return line.strip()
    return "No description"


def _parse_ast_from_file(pyfile: str) -> ast.Module:
    """
    Parse a Python file into an AST module object.

    Works on
    --------
    Local ``.py`` source files

    Parameters
    ----------
    pyfile : str
        Path to a Python file.

    Returns
    -------
    ast.Module
        Parsed AST tree.

    Examples
    --------
    >>>
    """
    with open(pyfile, "r", encoding="utf-8") as f:
        return ast.parse(f.read(), filename=pyfile)


def module_docstring_first_line_from_file(pyfile: str) -> str:
    """
    Return the first non-empty line of a module docstring without importing it.

    Works on
    --------
    Local ``.py`` source files

    Parameters
    ----------
    pyfile : str
        Path to a Python module file.

    Returns
    -------
    str
        One-line module summary, or ``"No description"``.

    Examples
    --------
    >>>
    """
    try:
        tree = _parse_ast_from_file(pyfile)
        return _first_line(ast.get_docstring(tree))
    except Exception:
        return "No description"


@dataclass(frozen=True)
class PublicSymbolSummary:
    """
    Summary record for a public symbol discovered via AST inspection.

    Attributes
    ----------
    name : str
        Public symbol name.
    kind : str
        Symbol kind (``"function"`` or ``"class"``).
    summary : str
        One-line docstring summary.
    """
    name: str
    kind: str   # "function" | "class"
    summary: str


def public_symbols_from_file(pyfile: str) -> List[PublicSymbolSummary]:
    """
    Return public functions and classes from a Python file using AST only.

    Works on
    --------
    Local ``.py`` source files

    Parameters
    ----------
    pyfile : str
        Path to a Python module file.

    Returns
    -------
    list[PublicSymbolSummary]
        Sorted public symbol summaries.

    Examples
    --------
    >>>
    """
    try:
        tree = _parse_ast_from_file(pyfile)
    except Exception:
        return []

    out: List[PublicSymbolSummary] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            if name.startswith("_"):
                continue
            doc = ast.get_docstring(node)
            out.append(PublicSymbolSummary(name=name, kind="function", summary=_first_line(doc)))

        elif isinstance(node, ast.ClassDef):
            name = node.name
            if name.startswith("_"):
                continue
            doc = ast.get_docstring(node)
            out.append(PublicSymbolSummary(name=name, kind="class", summary=_first_line(doc)))

    out.sort(key=lambda r: (r.kind, r.name.lower()))
    return out


# ---------------------------- folder scanning ----------------------------

def iter_py_files_recursive(
    root_dir: str,
    *,
    skip_private: bool = True,
    skip_init: bool = True,
    skip_dirs: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Collect Python files recursively from a root directory.

    Works on
    --------
    Local filesystem directory trees

    Parameters
    ----------
    root_dir : str
        Root directory to scan.
    skip_private : bool, optional
        If True, skip private files/directories prefixed with ``_``.
    skip_init : bool, optional
        If True, skip ``__init__.py`` files.
    skip_dirs : Iterable[str] or None, optional
        Directory names to exclude from recursion.

    Returns
    -------
    list[str]
        Sorted absolute paths to ``.py`` files.

    Examples
    --------
    >>>
    """
    root_dir = os.path.abspath(root_dir)
    skip_dirs = set(skip_dirs or {"__pycache__", ".git", ".venv", "venv", "site-packages", "dist", "build"})

    hits: List[str] = []
    for cur, dirs, files in os.walk(root_dir):
        # prune directories in-place
        dirs[:] = [d for d in dirs if d not in skip_dirs and not (skip_private and d.startswith("_"))]

        for fn in files:
            if not fn.endswith(".py"):
                continue
            if skip_init and fn == "__init__.py":
                continue
            if skip_private and fn.startswith("_"):
                continue
            hits.append(os.path.join(cur, fn))

    hits.sort(key=lambda p: p.lower())
    return hits


def list_modules_recursive_with_summaries(pkg_dir: str) -> List[Tuple[str, str]]:
    """
    List Python modules under a directory with one-line docstring summaries.

    Works on
    --------
    Local filesystem directory trees

    Parameters
    ----------
    pkg_dir : str
        Package directory to scan.

    Returns
    -------
    list[tuple[str, str]]
        ``(relative_path, module_summary)`` rows.

    Examples
    --------
    >>>
    """
    rows: List[Tuple[str, str]] = []
    for py in iter_py_files_recursive(pkg_dir):
        rel = os.path.relpath(py, pkg_dir).replace("\\", "/")
        summary = module_docstring_first_line_from_file(py)
        rows.append((rel, summary))
    return rows


# ---------------------------- resolving hints ----------------------------

def resolve_module_hint_to_file(module_hint: str) -> Optional[str]:
    """
    Resolve a dotted module name to a filesystem .py path using importlib spec.
    Returns None if it cannot be resolved.
    """
    try:
        spec = importlib.util.find_spec(module_hint)
        if spec and spec.origin and spec.origin.endswith(".py"):
            return spec.origin
    except Exception:
        return None
    return None
