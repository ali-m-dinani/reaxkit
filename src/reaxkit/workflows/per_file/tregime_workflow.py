"""
Temperature-regime (tregime) workflow for ReaxKit.

This workflow provides a utility for generating sample `tregime.in` files,
which define temperature schedules used in ReaxFF molecular dynamics
simulations.

It supports:
- Writing a correctly formatted `tregime.in` file with fixed-width columns.
- Controlling the number of temperature-regime rows written to the file.
- Automatically selecting a standardized output location when no explicit
  output path is provided.

The workflow is intended to simplify creation of valid temperature-regime
input files for testing, prototyping, and reproducible simulation setup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reaxkit.io.generators.tregime_generator import write_sample_tregime


def _task_generate(args: argparse.Namespace) -> int:
    # Default output location: reaxkit_outputs/tregime/tregime.in
    out_default = Path("reaxkit_outputs") / "tregime" / "tregime.in"

    # Prefer your project resolver if available
    out_path: Path
    if args.out:
        out_path = Path(args.out)
    else:
        try:
            from reaxkit.utils.path import resolve_output_path  # type: ignore

            out_path = Path(
                resolve_output_path(
                    kind="tregime",
                    out=args.out,
                    default_name="tregime.in",
                )
            )
        except Exception:
            out_path = out_default

    write_sample_tregime(out_path, n_rows=args.rows)

    print(f"[Done] Sample tregime generated in {out_path}")
    return 0


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    gen = subparsers.add_parser(
        "gen",
        help="Generate a sample tregime.in file (fixed-width columns).",
        description=(
            "Examples:\n"
            "  reaxkit tregime gen \n"
            "  reaxkit tregime gen --rows 5 \n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    gen.add_argument(
        "--out",
        default="reaxkit_generated_inputs/tregime.in",
        help=(
            "Output tregime filename or path. "
            "If not provided, writes to reaxkit_outputs/tregime/tregime.in"
        ),
    )
    gen.add_argument(
        "--rows",
        type=int,
        default=3,
        help="Number of sample rows to write.",
    )
    gen.set_defaults(_run=_task_generate)
