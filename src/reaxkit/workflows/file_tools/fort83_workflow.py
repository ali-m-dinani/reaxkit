"""Direct command workflow for extracting optimized force fields from fort.83."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    _ = command
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Extract the optimized force-field block from fort.83.\n\n"
        "Examples:\n"
        "  reaxkit extract-optimized-ffield --fort83 fort.83 --output ffield_optimized\n"
        "  reaxkit extract-optimized-ffield --output trained_ffield"
    )
    parser.add_argument("--fort83", default="fort.83", help="Path to fort.83")
    parser.add_argument("--output", default="ffield_optimized", help="Output path for the extracted force field")
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    _ = command
    fort83_path = Path(args.fort83)
    lines = fort83_path.read_text(encoding="utf-8").splitlines(keepends=True)

    start_index = None
    for idx, line in enumerate(lines):
        if "Error force field" in line:
            start_index = idx

    if start_index is None:
        print("[Warning] 'Error force field' not found in fort.83.")
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines[start_index + 1 :]), encoding="utf-8")
    print(f"[Done] Extracted content written to {out_path}")
    return 0
