# reaxkit/workflows/vregime_workflow.py
from __future__ import annotations

import argparse
from pathlib import Path

from reaxkit.io.vregime_generator import write_sample_vregime

def task_generate(args: argparse.Namespace) -> int:
    out_default = Path("reaxkit_outputs") / "vregime" / "vregime.in"

    if args.out:
        out_path = Path(args.out)
    else:
        # Prefer your project resolver if available
        try:
            from reaxkit.utils.path import resolve_output_path  # type: ignore

            out_path = Path(
                resolve_output_path(
                    kind="vregime",
                    out=None,
                    default_name="vregime.in",
                )
            )
        except Exception:
            out_path = out_default

    write_sample_vregime(out_path, n_rows=args.rows)

    print(f"[Done] Sample vregime generated in {out_path}")
    return 0


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "gen",
        help="Generate a sample vregime.in file (Volume regimes).",
        description=(
            "Examples:\n"
            "  reaxkit vregime gen \n"
            "  reaxkit vregime gen --rows 3 \n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--out",
        default="reaxkit_generated_inputs/vregime.in",
        help=(
            "Output vregime filename or path. "
            "If not provided, writes to reaxkit_outputs/vregime/vregime.in"
        ),
    )
    p.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of sample rows to write.",
    )
    p.set_defaults(_run=task_generate)
