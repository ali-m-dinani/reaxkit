"""
CLI workflow for introspecting ReaxKit modules and folders.

This workflow powers the `reaxkit intspec` command, allowing users to:
- Inspect a single Python module and view its top-level docstring summary
  along with public functions/classes and their one-line descriptions.
- Recursively scan a folder or package and list all contained `.py` files
  with their module docstring first lines.

It is designed as a lightweight discovery and navigation tool to help users
understand what functionality exists inside ReaxKit without opening files
manually.
"""


from __future__ import annotations

import argparse
import importlib
import os
from typing import List, Optional, Tuple

from tabulate import tabulate

from reaxkit.help.introspection_utils import (
    list_modules_recursive_with_summaries,
    module_docstring_first_line_from_file,
    public_symbols_from_file,
    resolve_module_hint_to_file,
)

# Try exact first, then common prefixes
CANDIDATE_PREFIXES: List[str] = [
    "",  # fully-qualified or exact
    "reaxkit.analysis.",
    "reaxkit.analysis.composed.",
    "reaxkit.analysis.per_file.",
    "reaxkit.workflows.",
    "reaxkit.workflows.per_file.",
    "reaxkit.workflows.composed.",
    "reaxkit.workflows.meta.",
    "reaxkit.io.",
    "reaxkit.io.handlers.",
    "reaxkit.io.generators.",
    "reaxkit.utils.",
    "reaxkit.utils.media.",
    "reaxkit.utils.numerical.",
    "reaxkit.help.",
]

# Canonical roots for folder short-hands like "workflow" / "analysis"
FOLDER_ROOTS: List[str] = [
    "reaxkit.analysis",
    "reaxkit.workflows",
    "reaxkit.io",
    "reaxkit.utils",
    "reaxkit.help",
]


def _print_table(rows: List[Tuple[str, str]], headers: Tuple[str, str]) -> int:
    if not rows:
        print("No items found.")
        return 0
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    return 0


def _resolve_pkg_dir_from_module_name(modname: str) -> Optional[str]:
    try:
        mod = importlib.import_module(modname)
        if getattr(mod, "__path__", None):
            return next(iter(mod.__path__))
        return None
    except Exception:
        return None


def _looks_like_path(hint: str) -> bool:
    return hint.endswith(".py") or ("/" in hint) or ("\\" in hint) or (os.sep in hint)


def _resolve_folder_hint_to_dir(folder_hint: str) -> Optional[str]:
    """
    Accepts:
      - filesystem directory (e.g., reaxkit/workflows)
      - dotted package (e.g., reaxkit.workflows)
      - shorthand (e.g., workflow/workflows, analysis, io, utils)
    """
    # Direct directory path
    if os.path.isdir(folder_hint):
        return os.path.abspath(folder_hint)

    # Dotted package
    d = _resolve_pkg_dir_from_module_name(folder_hint)
    if d:
        return d

    # Normalize singular/plural shorthand
    leafs = {folder_hint.strip()}
    if folder_hint.endswith("s"):
        leafs.add(folder_hint[:-1])
    else:
        leafs.add(folder_hint + "s")

    # Try mapping shorthand to known ReaxKit roots
    for root in FOLDER_ROOTS:
        for leaf in leafs:
            # e.g., reaxkit.workflows  (leaf == "workflow"/"workflows") → match root itself
            if leaf in {"workflow", "workflows"} and root == "reaxkit.workflows":
                d = _resolve_pkg_dir_from_module_name(root)
                if d:
                    return d

            # e.g., folder_hint="meta" → try reaxkit.workflows.meta
            d = _resolve_pkg_dir_from_module_name(f"{root}.{leaf}")
            if d:
                return d

    # Finally, try direct roots by their last component
    for root in FOLDER_ROOTS:
        d = _resolve_pkg_dir_from_module_name(root)
        if d and os.path.basename(d) in leafs:
            return d

    return None


def build_parser(p: argparse.ArgumentParser) -> None:
    """Define CLI args for `reaxkit intspec` (kind-level)."""
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", help="Module name (e.g. fort7_analyzer) or path to .py")
    g.add_argument("--folder", help="Folder/package (e.g. workflow, workflows, reaxkit/workflows)")


# ------------------------- FILE MODE -------------------------

def run_file(module_hint: str) -> int:
    """
    Show:
      - module docstring first line
      - table of public functions/classes with docstring first line
    """
    pyfile: Optional[str] = None

    # 1) direct path
    if _looks_like_path(module_hint):
        if os.path.isfile(module_hint):
            pyfile = os.path.abspath(module_hint)
        else:
            print(f"File not found: {module_hint}")
            return 1
    else:
        # 2) fully-qualified or prefixed module name resolution
        tried: List[str] = []
        for prefix in CANDIDATE_PREFIXES:
            name = prefix + module_hint
            tried.append(name)
            pyfile = resolve_module_hint_to_file(name)
            if pyfile:
                break

        if not pyfile:
            print("Could not resolve module to a .py file. Tried:")
            for t in tried:
                print(f"  - {t}")
            return 1

    # Print module docstring first line
    mod_sum = module_docstring_first_line_from_file(pyfile)
    print(f"\nModule: {module_hint}")
    print(f"File:   {pyfile}")
    print(f"Doc:    {mod_sum}\n")

    # Print public symbols table
    syms = public_symbols_from_file(pyfile)
    rows = [(s.kind, s.name, s.summary) for s in syms]
    if not rows:
        print("No public functions/classes found.")
        return 0

    print(tabulate(rows, headers=("Kind", "Name", "Summary"), tablefmt="grid"))
    return 0


# ------------------------- FOLDER MODE -------------------------

def run_folder(folder_hint: str) -> int:
    """
    Recursively list all .py files under the folder/package with module docstring first line.
    """
    pkg_dir = _resolve_folder_hint_to_dir(folder_hint)
    if not pkg_dir:
        print(f"Could not resolve folder '{folder_hint}' to a package directory.")
        return 1

    rows = list_modules_recursive_with_summaries(pkg_dir)
    # Use relative path as the "name" so nested folders are visible
    return _print_table(rows, headers=("Module (.py under folder)", "Docstring (1st line)"))


# ------------------------- main glue -------------------------

def run_main(file: str | None, folder: str | None) -> int:
    """
    - If --file: show module docstring + public symbol summary table
    - If --folder: recursively list .py files + module docstrings
    """
    if bool(file) == bool(folder):
        print("Please pass exactly one of: --file <module> OR --folder <package/folder>")
        return 2
    if file:
        return run_file(file)
    return run_folder(folder)  # type: ignore[arg-type]


def register_tasks(subparsers) -> None:
    """Task-level entry: `reaxkit intspec run ...`."""
    p = subparsers.add_parser(
        "run",
        help="Introspect a module (--file) or folder (--folder).",
        description=(
            "Introspect a module (--file) or folder (--folder).\n"
            "Examples:\n"
            "  reaxkit intspec --folder workflow\n"
            "  reaxkit intspec run --folder workflow\n"
            "  reaxkit intspec --file fort7_analyzer\n"
            "  reaxkit intspec run --file fort7_analyzer\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", help="Module name (e.g. fort7_analyzer) or path to .py")
    g.add_argument("--folder", help="Folder/package (e.g. workflow, workflows, reaxkit/workflows)")
    p.set_defaults(_run=lambda args: run_main(getattr(args, "file", None), getattr(args, "folder", None)))