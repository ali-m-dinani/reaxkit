# reaxkit/workflows/params_workflow.py
from __future__ import annotations

import argparse

from reaxkit.io.params_handler import ParamsHandler
from reaxkit.io.ffield_handler import FFieldHandler

from reaxkit.analysis.params_analyzer import get_params, interpret_params
from reaxkit.utils.path import resolve_output_path


def task_get(args: argparse.Namespace) -> int:
    params_handler = ParamsHandler(args.file)

    if args.interpret:
        # Interpreted params require ffield
        ffield_handler = FFieldHandler(args.ffield)
        df = interpret_params(
            params_handler=params_handler,
            ffield_handler=ffield_handler,
            add_term=(not args.no_term),
        )

        # Match default "drop duplicates" behavior to raw get()
        if not args.keep_duplicates:
            df = df.drop_duplicates(
                subset=["ff_section", "ff_section_line", "ff_parameter"],
                keep="first",
            )
    else:
        df = get_params(
            params_handler,
            sort_by=None,  # no sorting by default (handled below if user requests)
            ascending=True,
            drop_duplicate=(not args.keep_duplicates),
        )

    # Optional sorting (default is none)
    if args.sort_by:
        if args.sort_by not in df.columns:
            raise SystemExit(
                f"❌ sort-by column '{args.sort_by}' not found. Available: {', '.join(df.columns)}"
            )
        df = df.sort_values(by=args.sort_by, ascending=(not args.descending))

    # Export or preview
    if args.export:
        out = resolve_output_path(args.export, workflow="params")
        df.to_csv(out, index=False)
        print(f"[Done] Exported the requested data to {out}")
    else:
        print(df.head(20).to_string(index=False))

    return 0

#####################################################################################

def _add_common_params_io_args(p: argparse.ArgumentParser) -> None:
    # Core IO
    p.add_argument("--file", default="params", help="Path to params file.")
    p.add_argument("--export", default=None, help="Path to export CSV data.")

    # Default behavior requested:
    # - remove duplicates by default
    # - no sorting by default
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
        help="If set, interpret params pointers into the ffield (adds section/row/param/value/term columns).",
    )
    p.add_argument(
        "--ffield",
        default="ffield",
        help="Path to ffield file (required when --interpret is set).",
    )
    p.add_argument(
        "--no-term",
        action="store_true",
        help="If set, do not build readable term (e.g., C-C-H) during interpretation.",
    )

    p.set_defaults(_run=task_get)
