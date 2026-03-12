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
from reaxkit.engine.reaxff.generators.xmolout_generator import write_xmolout_from_handler
from reaxkit.engine.reaxff.io.xmolout_handler import XmoloutHandler


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    _ = command
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Trim xmolout to a lighter file with only atom type and x,y,z coordinates.\n\n"
        "Examples:\n"
        "  reaxkit trim-xmolout --file xmolout --output xmolout_trimmed\n"
        "  reaxkit trim-xmolout --output xmolout_light"
    )
    parser.add_argument("--file", default="xmolout", help="Input xmolout file")
    parser.add_argument("--output", default="xmolout_trimmed", help="Output trimmed xmolout file")
    parser.add_argument("--copy-to-dot", action="store_true", help="Also copy generated output to current directory")
    add_storage_cli_arguments(parser)
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    out_path, layout = prepare_generator_output(args, command=command, output_value=str(args.output))
    xh = XmoloutHandler(args.file)
    write_xmolout_from_handler(xh, out_path, include_extras=False)
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
