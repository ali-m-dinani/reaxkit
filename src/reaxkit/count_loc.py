"""Count lines of code in a Python project.

Usage (from inside the reaxkit/ directory):
    python count_loc.py
Optional:
    python count_loc.py --root . --out loc_report.csv
"""

from __future__ import annotations
import argparse
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
    s = line.strip()
    return (not s) or s.startswith("#")

def count_code_lines(py_path: Path) -> int:
    """Count code lines (non-blank, non-comment) in a .py file."""
    try:
        with py_path.open("r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for line in f if not is_comment_or_blank(line))
    except Exception:
        # If a file can't be read for any reason, treat as 0 LOC but keep going.
        return 0

def iter_python_files(root: Path, excludes: set[str]) -> list[Path]:
    files = []
    for p in root.rglob("*.py"):
        # Skip excluded directories anywhere in the path
        if any(part in excludes for part in p.parts):
            continue
        files.append(p)
    return files

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively count Python LOC and export CSV (file,loc)."
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
        help="Count all physical lines (includes blanks and comments).",
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

    rows = []
    total_loc = 0

    for f in py_files:
        if args.include_comments:
            try:
                with f.open("r", encoding="utf-8", errors="ignore") as fh:
                    loc = sum(1 for _ in fh)  # physical lines
            except Exception:
                loc = 0
        else:
            loc = count_code_lines(f)

        rel = f.relative_to(root)
        rows.append((str(rel), loc))
        total_loc += loc

    # Write CSV: Sort by LOC (descending)
    rows.sort(key=lambda x: x[1], reverse=True)

    # Write CSV sorted by LOC
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "loc"])
        writer.writerows(rows)

    # Print summary
    print(f"\nWrote {len(rows)} files to: {out_csv}")
    print(f"Total lines of code: {total_loc:,}")
    print("\nTop 6 files by LOC:")
    for file, loc in rows[:6]:
        print(f"  {file:<40} {loc}")

if __name__ == "__main__":
    main()
