"""
fort.83 post-processing workflow for ReaxKit.

This workflow provides a lightweight utility for processing ReaxFF `fort.83`
files, which typically record force-field optimization progress and error
information during training or fitting runs.

It supports:
- Locating the final occurrence of the marker line `Error force field` in a
  `fort.83` file.
- Extracting all subsequent lines, which usually correspond to the optimized
  force-field parameter block.
- Writing the extracted content to a new output file for reuse, inspection,
  or archiving.

The workflow is intended to streamline recovery of optimized force-field
parameters after ReaxFF training runs.
"""


import argparse


def _fort83_update_task(args: argparse.Namespace) -> int:
    """
    Reads fort.83, finds last occurrence of 'Error force field',
    and exports all lines after it to output file.
    """
    with open(args.file, "r") as f:
        lines = f.readlines()

    start_index = None
    for i, line in enumerate(lines):
        if "Error force field" in line:
            start_index = i

    if start_index is None:
        print("[Warning] 'Error force field' not found in file.")
        return 1

    extracted_lines = lines[start_index + 1:]

    with open(args.export, "w") as out:
        out.writelines(extracted_lines)

    print(f"[Done] Extracted content written to {args.export}")
    return 0


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    """
    Directly register 'get' under fort83 without extra nesting.
    """
    p = subparsers.add_parser(
        "update",
        help="Extract all lines after last 'Error force field' in fort.83",
    )
    p.add_argument("--file", default="fort.83", help="Path to fort.83 file")
    p.add_argument("--export", default="ffield_optimized", help="Output file name")
    p.set_defaults(_run=_fort83_update_task)
