"""Count lines of code in a Python project.

Usage (from inside your project directory):
    python count_loc.py
Optional:
    python count_loc.py --root . --out loc_report.csv
"""

from __future__ import annotations
import argparse
import ast
import csv
from pathlib import Path

# Folders to skip during the walk
DEFAULT_EXCLUDES = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
    ".idea",
    ".vscode",
    "site-packages",
    "node_modules",
}


def is_comment_or_blank(line: str) -> bool:
    """
    Legacy helper: returns True if the line is blank or a comment-only line.
    Kept for backward compatibility but not used in main logic.
    """
    s = line.strip()
    return (not s) or s.startswith("#")


def iter_python_files(root: Path, excludes: set[str]) -> list[Path]:
    """Recursively find all .py files under root, skipping excluded dirs."""
    files = []
    for p in root.rglob("*.py"):
        # Skip excluded directories anywhere in the path
        if any(part in excludes for part in p.parts):
            continue
        files.append(p)
    return files


def _docstring_line_numbers(source: str, filename: str) -> set[int]:
    """
    Return a set of line numbers (1-based) that belong to *docstrings*
    (module, class, function/async function) based on the AST.
    """
    doc_lines: set[int] = set()
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        # If the file is not valid Python (or incomplete), treat everything
        # as non-docstring.
        return doc_lines

    # Module-level and all function/class nodes
    targets = [tree]
    targets.extend(
        node
        for node in ast.walk(tree)
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        )
    )

    for node in targets:
        # ast.get_docstring tells us whether a docstring exists
        if ast.get_docstring(node, clean=False) is None:
            continue

        # Docstring must be the first statement in the body
        body = getattr(node, "body", [])
        if not body:
            continue
        expr0 = body[0]
        if not isinstance(expr0, ast.Expr):
            continue

        value = expr0.value
        # Py3.8+: docstring is usually ast.Constant; older: ast.Str
        lineno = getattr(value, "lineno", None)
        end_lineno = getattr(value, "end_lineno", None) or lineno
        if lineno is None:
            continue

        for ln in range(lineno, end_lineno + 1):
            doc_lines.add(ln)

    return doc_lines


def classify_python_file(py_path: Path) -> dict[str, int]:
    """
    Classify each line in a .py file as code, comment, docstring, or blank.

    Returns a dict with keys:
        - code
        - comments
        - docstrings
        - blank
        - total
    """
    try:
        text = py_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        # If unreadable, treat as 0 for all categories.
        return {"code": 0, "comments": 0, "docstrings": 0, "blank": 0, "total": 0}

    lines = text.splitlines()
    docstring_lines = _docstring_line_numbers(text, str(py_path))

    code = comments = docstrings = blank = 0

    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()

        if not stripped:
            blank += 1
        elif lineno in docstring_lines:
            # Mark docstring lines first to avoid misclassifying them
            # as comments or code.
            docstrings += 1
        elif stripped.startswith("#"):
            # Comment-only line
            comments += 1
        else:
            # Everything else is treated as code (includes regular strings,
            # inline comments, etc.).
            code += 1

    total = code + comments + docstrings + blank
    return {
        "code": code,
        "comments": comments,
        "docstrings": docstrings,
        "blank": blank,
        "total": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively count Python LOC and export CSV with per-file breakdown "
            "(code, comments, docstrings, blank)."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory to scan (default: current directory).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="loc_report.csv",
        help="Output CSV filename (default: loc_report.csv).",
    )
    parser.add_argument(
        "--include-comments",
        action="store_true",
        help=(
            "Count all physical lines as LOC metric (code + comments + docstrings + blank). "
            "By default, only code lines are used for the LOC metric."
        ),
    )
    parser.add_argument(
        "--extra-exclude",
        action="append",
        default=[],
        help="Extra folder name(s) to exclude (can be used multiple times).",
    )

    args = parser.parse_args()
    root = Path(args.root).resolve()
    out_csv = Path(args.out).resolve()
    excludes = set(DEFAULT_EXCLUDES) | set(args.extra_exclude)

    if not root.exists():
        raise SystemExit(f"Root path not found: {root}")

    py_files = iter_python_files(root, excludes)
    py_files.sort()

    csv_rows = []
    total_loc_metric = 0  # what we print as "Total lines of code" or "Total counted lines"

    # Global totals for summary
    global_counts = {"code": 0, "comments": 0, "docstrings": 0, "blank": 0, "total": 0}

    # Index to sort by: code (1) or total (5)
    sort_index = 5 if args.include_comments else 1

    for f in py_files:
        counts = classify_python_file(f)

        # Update global totals
        for key in global_counts:
            global_counts[key] += counts[key]

        # Metric used for totals & ranking
        loc_metric = counts["total"] if args.include_comments else counts["code"]
        total_loc_metric += loc_metric

        rel = f.relative_to(root)
        csv_rows.append(
            (
                str(rel),
                counts["code"],
                counts["comments"],
                counts["docstrings"],
                counts["blank"],
                counts["total"],
            )
        )

    # Sort CSV rows by chosen metric (code or total)
    csv_rows.sort(key=lambda row: row[sort_index], reverse=True)

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "code", "comments", "docstrings", "blank", "total"])
        writer.writerows(csv_rows)

    # Print metric summary
    metric_label = (
        "Total counted lines (code + comments + docstrings + blank)"
        if args.include_comments
        else "Total lines of code (excluding comments/docstrings/blanks)"
    )

    print(f"\nWrote {len(csv_rows)} files to: {out_csv}")
    print(f"{metric_label}: {total_loc_metric:,}\n")

    # Print top 6 files
    print("Top 6 files by this metric:")
    for file, code, comments, docs, blank, total in csv_rows[:6]:
        loc_metric = total if args.include_comments else code
        print(f"  {file:<40} {loc_metric:,}")
    print()

    # --- Global summary by line type with percentages ---
    total_lines = global_counts["total"] or 1  # avoid division by zero

    def fmt(name: str, count: int) -> str:
        pct = (count / total_lines) * 100.0
        return f"{name:<25} {count:>8,} ({pct:>3.0f}%)"

    print("Summary by Line Type:")
    print(fmt("Total lines of code:", global_counts["code"]))
    print(fmt("Total comment lines:", global_counts["comments"]))
    print(fmt("Total docstring lines:", global_counts["docstrings"]))
    print(fmt("Total blank lines:", global_counts["blank"]))
    print("-" * 50)
    print(f"{'Total physical lines:':<25} {global_counts['total']:>8,}")
    print()


if __name__ == "__main__":
    main()
