"""Direct command workflow for vregime generation."""

from __future__ import annotations

import argparse

from reaxkit.cli.path import resolve_output_path
from reaxkit.engine.reaxff.generators.vregime_generator import write_sample_vregime


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    _ = command
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Generate a sample vregime.in file.\n\n"
        "Examples:\n"
        "  reaxkit make-vregime\n"
        "  reaxkit make-vregime --rows 5 --output vregime.in"
    )
    parser.add_argument("--output", default="vregime.in", help="Output vregime path")
    parser.add_argument("--rows", type=int, default=5, help="Number of sample rows")
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    _ = command
    out = resolve_output_path(args.output, workflow="vregime")
    write_sample_vregime(out, n_rows=args.rows)
    print(f"[Done] Sample vregime generated in {out}")
    return 0
