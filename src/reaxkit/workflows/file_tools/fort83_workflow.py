"""Direct command workflow for extracting optimized force fields from fort.83."""

from __future__ import annotations

import argparse
from pathlib import Path

from reaxkit.core.generator_runtime import (
    maybe_copy_output_to_dot,
    persist_generator_metadata,
    prepare_generator_output,
    print_saved_dirs,
)
from reaxkit.core.storage_layout import add_storage_cli_arguments


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    _ = command
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Extract the optimized force-field block from a `fort.83` file.\n"
        "This command scans for the `Error force field` marker and writes everything after that\n"
        "marker to a separate output file. Use it to recover the trained/optimized force field\n"
        "from optimization output.\n\n"
        "Examples:\n"
        "  1. Extract from a specific `fort.83` file with explicit output name:\n"
        "   reaxkit extract-optimized-ffield --fort83 fort.83 --output ffield_optimized\n\n"
        "  2. Extract using a custom output filename:\n"
        "   reaxkit extract-optimized-ffield --output trained_ffield"
    )
    parser.add_argument(
        "--fort83",
        default="fort.83",
        help="Path to fort.83. Example: --fort83 runs/job1/fort.83, which reads that optimization output file.",
    )
    parser.add_argument(
        "--output",
        default="ffield_optimized",
        help="Output path for the extracted force field. Example: --output trained_ffield, which writes extracted parameters to that filename.",
    )
    parser.add_argument(
        "--copy-to-dot",
        action="store_true",
        help="Also copy generated output to current directory. Example: --copy-to-dot, which keeps a convenience copy where you run the command.",
    )
    add_storage_cli_arguments(parser)
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    out_path, layout = prepare_generator_output(args, command=command, output_value=str(args.output))
    fort83_path = Path(args.fort83)
    lines = fort83_path.read_text(encoding="utf-8").splitlines(keepends=True)

    start_index = None
    for idx, line in enumerate(lines):
        if "Error force field" in line:
            start_index = idx

    if start_index is None:
        print("[Warning] 'Error force field' not found in fort.83.")
        return 1

    out_path.write_text("".join(lines[start_index + 1 :]), encoding="utf-8")
    persist_generator_metadata(
        args,
        command=command,
        output_path=out_path,
        layout=layout,
        copy_to_dot=bool(getattr(args, "copy_to_dot", False)),
    )
    copied = maybe_copy_output_to_dot(out_path, enabled=bool(getattr(args, "copy_to_dot", False)))
    dirs = [out_path.parent]
    if copied is not None:
        dirs.append(copied.parent)
    print_saved_dirs(dirs)
    return 0
