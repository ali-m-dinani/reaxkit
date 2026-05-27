"""Direct command workflow for xmolout file utilities."""

from __future__ import annotations

import argparse

from reaxkit.core.generator_runtime import (
    maybe_copy_output_to_dot,
    persist_generator_metadata,
    prepare_generator_output,
    print_saved_dirs,
)
from reaxkit.core.storage_layout import add_storage_cli_arguments
from reaxkit.engine.reaxff.generators.xmolout_generator import trim_xmolout

ALL_COMMANDS = ("trim-xmolout",)
ALL_LEGACY_COMMANDS = ("trim_xmolout",)


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    _ = command
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Trim an `xmolout` file to a lighter format containing only atom type and x/y/z coordinates.\n"
        "This command is useful when full xmolout output is too large for quick inspection or downstream\n"
        "tools that support reading atomic positions (not other variables like velocities, molecule numbers, etc."
        "which may be written to the xmolout file using ixmolo keyword in the control file)."
        " It writes a reduced output file and does not modify\n"
        "the input file in place.\n\n"
        "Examples:\n"
        "  1. Trim a specific xmolout file to a named output:\n"
        "   reaxkit trim-xmolout --file xmolout --output xmolout_trimmed\n\n"
        "  2. Trim using a custom output filename:\n"
        "   reaxkit trim-xmolout --output xmolout_light"
    )
    parser.add_argument(
        "--file",
        default="xmolout",
        help="Input xmolout file. Example: --file runs/job1/xmolout, which reads that trajectory/output file as input.",
    )
    parser.add_argument(
        "--output",
        default="xmolout_trimmed",
        help="Output trimmed xmolout file. Example: --output xmolout_light, which writes the reduced-content file with that name.",
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
    trim_xmolout(input_file=args.file, out_path=out_path)
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
