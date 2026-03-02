"""Params-file workflow for ReaxKit."""

from __future__ import annotations

import argparse

from reaxkit.analysis.params import (
    ForceFieldOptimizationParameterRequest,
    ForceFieldOptimizationParameterTask,
)
from reaxkit.cli.path import resolve_output_path
from reaxkit.domain.data_models import ForceFieldOptimizationParameterData, ForceFieldParametersData
from reaxkit.engine.reaxff.adapter import ReaxFFAdapter


def _task_get(args: argparse.Namespace) -> int:
    adapter = ReaxFFAdapter()
    params_data = adapter.load(
        ForceFieldOptimizationParameterData,
        {"params": args.file, "input": args.file},
    )

    force_field = None
    if args.interpret:
        force_field = adapter.load(
            ForceFieldParametersData,
            {"ffield": args.ffield, "input": args.ffield},
        )

    df = ForceFieldOptimizationParameterTask().run(
        params_data,
        ForceFieldOptimizationParameterRequest(
            sort_by=args.sort_by,
            ascending=(not args.descending),
            drop_duplicate=(not args.keep_duplicates),
            interpret=bool(args.interpret),
            force_field=force_field,
            add_term=(not args.no_term),
        ),
    ).table

    if args.export:
        out = resolve_output_path(args.export, workflow="params")
        df.to_csv(out, index=False)
        print(f"[Done] Exported the requested data to {out}")
    else:
        print(df.head(20).to_string(index=False))
    return 0


def _add_common_params_io_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--file", default="params", help="Path to params file.")
    p.add_argument("--export", default=None, help="Path to export CSV data.")
    p.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="If set, do NOT drop duplicates (default drops duplicates).",
    )
    p.add_argument(
        "--sort-by",
        default=None,
        help="Optional column name to sort by (default: no sorting).",
    )
    p.add_argument(
        "--descending",
        action="store_true",
        help="If set, sort in descending order (only if --sort-by is used).",
    )


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "get",
        help="Load params table (optionally interpret pointers into ffield)",
        description=(
            "Examples:\n"
            "  reaxkit params get --export params.csv\n"
            "\n"
            "Interpreted params:\n"
            "  reaxkit params get --interpret --export params_interpreted.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    _add_common_params_io_args(p)
    p.add_argument(
        "--interpret",
        action="store_true",
        help="Interpret params pointers into the ffield.",
    )
    p.add_argument(
        "--ffield",
        default="ffield",
        help="Path to ffield file (required when --interpret is set).",
    )
    p.add_argument(
        "--no-term",
        action="store_true",
        help="Do not build readable term labels during interpretation.",
    )
    p.set_defaults(_run=_task_get)
