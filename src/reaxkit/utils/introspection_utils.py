"""Tools for extracting one-line summaries from modules and their functions for documentation generation."""

import os
import ast
import inspect
from typing import List, Tuple, Any

def get_function_summaries(module: Any) -> List[Tuple[str, str]]:
    """
    Return a list of (function_name, first_line_of_docstring) for all public
    functions defined *in this module* (skip imported ones).
    """
    rows: List[Tuple[str, str]] = []
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_"):
            continue
        if getattr(func, "__module__", None) != module.__name__:
            continue
        doc = inspect.getdoc(func)
        summary = doc.splitlines()[0].strip() if doc else "No description"
        rows.append((name, summary))
    rows.sort(key=lambda x: x[0].lower())
    return rows

def _first_line(s: str | None) -> str:
    if not s:
        return "No description"
    for line in s.strip().splitlines():
        if line.strip():
            return line.strip()
    return "No description"

def get_module_docstring_first_line(pyfile: str) -> str:
    """Return the first non-empty line of a module's docstring without importing."""
    try:
        with open(pyfile, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=pyfile)
        return _first_line(ast.get_docstring(tree))
    except Exception:
        return "No description"

def list_package_modules_with_summaries(pkg_dir: str) -> List[Tuple[str, str]]:
    """
    Given a package directory, return [(module_name, one_line_summary), ...]
    for each top-level .py in that directory (excluding __init__.py and private files).
    """
    rows: List[Tuple[str, str]] = []
    for fn in sorted(os.listdir(pkg_dir)):
        if not fn.endswith(".py"):
            continue
        if fn == "__init__.py" or fn.startswith("_"):
            continue
        mod_name = os.path.splitext(fn)[0]
        summary = get_module_docstring_first_line(os.path.join(pkg_dir, fn))
        rows.append((mod_name, summary))
    return rows
