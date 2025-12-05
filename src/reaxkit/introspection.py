"""Introspect modules or packages to understand what is available in a folder or a script.

Examples:
  # Functions in a specific file/module
  reaxkit intspec run --file xmolout_workflow
  reaxkit intspec --file reaxkit/workflows/xmolout_workflow.py

  # One-line summaries for every module in a folder
  reaxkit intspec run --folder workflow
  reaxkit intspec --folder reaxkit/workflows

"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import List, Tuple, Optional
from tabulate import tabulate

from .utils.introspection_utils import (
    get_function_summaries,
    list_package_modules_with_summaries,
)

# Try exact first, then common prefixes (analysis before workflows).
CANDIDATE_PREFIXES: List[str] = [
    "",                       # exact / fully-qualified / bare
    "reaxkit.analysis.",
    "reaxkit.workflows.",
]

# For folder short-hands like "workflow", try mapping to common package names.
FOLDER_CANONICAL: List[str] = [
    "reaxkit.analysis",
    "reaxkit.workflows",
]

def _looks_like_path(hint: str) -> bool:
    """Heuristic: treat as a path if it ends with .py or contains a path separator."""
    return hint.endswith(".py") or ("/" in hint) or ("\\" in hint) or (os.sep in hint)

def _print_rows(rows: List[Tuple[str, str]], headers=("Name", "Summary")) -> int:
    if not rows:
        print("No items found.")
        return 0
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    return 0

# ------------------------- FILE/MODULE MODE -------------------------

def _import_by_path(path: str):
    """Import a module from a filesystem path like 'reaxkit/workflows/x.py'."""
    abspath = os.path.abspath(path)
    if not os.path.exists(abspath):
        raise FileNotFoundError(abspath)
    if os.path.isdir(abspath):
        # If a directory is given, import its __init__.py
        init_py = os.path.join(abspath, "__init__.py")
        if not os.path.exists(init_py):
            raise ImportError(f"No __init__.py in package directory: {abspath}")
        mod_name = os.path.basename(abspath)
        spec = importlib.util.spec_from_file_location(mod_name, init_py)
    else:
        mod_name = os.path.splitext(os.path.basename(abspath))[0]
        spec = importlib.util.spec_from_file_location(mod_name, abspath)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {abspath}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod.__name__] = mod
    spec.loader.exec_module(mod)
    return mod

def run_file(module_hint: str) -> int:
    """
    Import a module from hint (dotted name or path), list functions + first-line docstrings.
    """
    tried = []
    last_err: Exception | None = None

    # 1) Path import
    if _looks_like_path(module_hint):
        try:
            mod = _import_by_path(module_hint)
            rows = get_function_summaries(mod)
            return _print_rows(rows, headers=("Function", "Summary"))
        except Exception as e:
            last_err = e
            tried.append(f"[path] {module_hint}")

    # 2) Fully-qualified module
    try:
        mod = importlib.import_module(module_hint)
        rows = get_function_summaries(mod)
        return _print_rows(rows, headers=("Function", "Summary"))
    except Exception as e:
        last_err = e
        tried.append(f"[module] {module_hint}")

    # 3) Common prefixes
    for prefix in CANDIDATE_PREFIXES:
        fullname = prefix + module_hint
        try:
            mod = importlib.import_module(fullname)
            rows = get_function_summaries(mod)
            return _print_rows(rows, headers=("Function", "Summary"))
        except Exception as e:
            last_err = e
            tried.append(f"[module] {fullname}")

    print(
        "Could not import the target. Tried:\n  - "
        + "\n  - ".join(tried)
        + (f"\nLast error: {last_err}" if last_err else "")
    )
    return 1

# ------------------------- FOLDER/PACKAGE MODE -------------------------

def _resolve_pkg_dir_from_module_name(modname: str) -> Optional[str]:
    try:
        mod = importlib.import_module(modname)
        if getattr(mod, "__path__", None):
            return next(iter(mod.__path__))
        # Not a package
        return None
    except Exception:
        return None

def _resolve_folder_hint_to_dir(folder_hint: str) -> Optional[str]:
    """
    Resolve a 'folder' hint into a filesystem directory of a Python package.
    Accepts:
      - real filesystem paths (e.g., reaxkit/workflows)
      - dotted packages (e.g., reaxkit.workflows)
      - short-hands like 'workflow' or 'workflows' → tried under common parents
    """
    # Direct filesystem path
    if os.path.isdir(folder_hint):
        return os.path.abspath(folder_hint)

    # Dotted package name
    d = _resolve_pkg_dir_from_module_name(folder_hint)
    if d:
        return d

    # Try common parents with the given leaf name (normalize singular/plural)
    leafs = {folder_hint}
    if folder_hint.endswith("s"):
        leafs.add(folder_hint[:-1])
    else:
        leafs.add(folder_hint + "s")

    for parent in FOLDER_CANONICAL:
        for leaf in leafs:
            d = _resolve_pkg_dir_from_module_name(f"{parent}.{leaf}")
            if d:
                return d

    # Also try top-level canonical packages directly
    for parent in FOLDER_CANONICAL:
        d = _resolve_pkg_dir_from_module_name(parent)
        if d and os.path.basename(d) in leafs:
            return d

    return None

def run_folder(folder_hint: str) -> int:
    """
    List immediate .py modules in the given package/folder with one-line module summaries.
    """
    pkg_dir = _resolve_folder_hint_to_dir(folder_hint)
    if not pkg_dir:
        print(f"Could not resolve folder '{folder_hint}' to a package directory.")
        return 1
    rows = list_package_modules_with_summaries(pkg_dir)
    return _print_rows(rows, headers=("Module", "Summary"))

# ------------------------- CLI GLUE -------------------------

def run_main(file: str | None, folder: str | None) -> int:
    """
    Dispatcher used by both 'run' subcommand and the default (no-subcommand) form.
    Exactly one of file/folder should be provided.
    """
    if bool(file) == bool(folder):
        print("Please pass exactly one of: --file <module>  OR  --folder <package>")
        return 2
    if file:
        return run_file(file)
    return run_folder(folder)  # type: ignore[arg-type]

def register_tasks(subparsers):
    """
    Wire into the top-level CLI:
      reaxkit intspec run --file <module_or_path>
      reaxkit intspec run --folder <package_or_dir>

    If your top-level CLI wants to allow the default action without 'run', add
    the same two arguments to the parent 'intspec' parser and set:
      intspec_parser.set_defaults(_run=lambda args: run_main(args.file, args.folder))
    """
    p = subparsers.add_parser(
        "run",
        help=("List functions for a module (--file), or list submodules if a "
              "package/folder is given (--folder)."),
    )

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--file",
        help=("Module (e.g., xmolout_workflow or reaxkit.workflows.xmolout_workflow) "
              "or path (e.g., reaxkit/workflows/xmolout_workflow.py).")
    )
    g.add_argument(
        "--folder",
        help=("Package name (e.g., workflows or reaxkit.workflows) or a directory "
              "(e.g., reaxkit/workflows).")
    )
    p.set_defaults(_run=lambda args: run_main(args.file, args.folder))
