"""Direct command workflow for xmolout file utilities."""

from __future__ import annotations

import argparse

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
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    _ = command
    xh = XmoloutHandler(args.file)
    write_xmolout_from_handler(xh, args.output, include_extras=False)
    print(f"[Done] Wrote trimmed xmolout (type + x,y,z only) to {args.output}")
    return 0
