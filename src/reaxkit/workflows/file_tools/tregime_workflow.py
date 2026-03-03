"""Direct command workflow for tregime generation."""

from __future__ import annotations

import argparse

from reaxkit.cli.path import resolve_output_path
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
    parser.add_argument("--rows", type=int, default=3, help="Number of sample rows")
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    _ = command
    out = resolve_output_path(args.output, workflow="tregime")
    write_sample_tregime(out, n_rows=args.rows)
    print(f"[Done] Sample tregime generated in {out}")
    return 0
