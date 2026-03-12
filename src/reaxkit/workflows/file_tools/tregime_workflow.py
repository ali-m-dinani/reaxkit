"""Direct command workflow for tregime generation."""

from __future__ import annotations

import argparse

from reaxkit.core.generator_runtime import (
    maybe_copy_output_to_dot,
    persist_generator_metadata,
    prepare_generator_output,
    print_saved_dirs,
)
from reaxkit.core.storage_layout import add_storage_cli_arguments
from reaxkit.engine.reaxff.generators.tregime_generator import write_sample_tregime


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    _ = command
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Generate a sample tregime.in file.\n\n"
        "Examples:\n"
        "  reaxkit make-tregime\n"
        "  reaxkit make-tregime --rows 5 --output tregime.in"
    )
    parser.add_argument("--output", default="tregime.in", help="Output tregime path")
    parser.add_argument("--copy-to-dot", action="store_true", help="Also copy generated output to current directory")
    parser.add_argument("--rows", type=int, default=3, help="Number of sample rows")
    add_storage_cli_arguments(parser)
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    out, layout = prepare_generator_output(args, command=command, output_value=str(args.output))
    write_sample_tregime(out, n_rows=args.rows)
    persist_generator_metadata(
        args,
        command=command,
        output_path=out,
        layout=layout,
        copy_to_dot=bool(getattr(args, "copy_to_dot", False)),
    )
    copied = maybe_copy_output_to_dot(out, enabled=bool(getattr(args, "copy_to_dot", False)))
    dirs = [out.parent]
    if copied is not None:
        dirs.append(copied.parent)
    print_saved_dirs(dirs)
    return 0
