"""
Interactive help and discovery workflow for ReaxKit.

This workflow supports two complementary help modes:
1) capability discovery (intent -> responsible analysis tasks/workflows/generators)
2) ReaxFF file discovery (query -> relevant input/output files)
"""

from __future__ import annotations

import argparse


def build_parser(p: argparse.ArgumentParser) -> None:
    """Define CLI arguments for `reaxkit help`."""
    p.formatter_class = argparse.RawTextHelpFormatter
    p.description = (
        "Interactive help and discovery for ReaxKit commands and file semantics.\n\n"
        "Examples:\n"
        "  reaxkit help \"msd\"\n"
        "  reaxkit help \"bond order\"\n"
        "  reaxkit help \"bond order\" --top 3\n"
        "  reaxkit help \"restraint\" --engine reaxff\n"
        "  reaxkit help \"fort.7\" --all-info\n"
        "  reaxkit help \"xmolout\" --all-info"
    )

    p.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Search query (use quotes for multi-word queries).",
    )

    p.add_argument(
        "--top",
        type=int,
        default=8,
        help="Maximum results per category (generator/file/analyzer/workflow), sorted by score.",
    )

    p.add_argument(
        "--engine",
        type=str,
        default=None,
        help="Optional engine context (example: reaxff) for dataclass-to-file mappings.",
    )

    p.add_argument(
        "--all-info",
        dest="all_info",
        action="store_true",
        help="Show detailed implementation and file/dataclass/analyzer mapping information.",
    )

    p.add_argument(
        "--exact-match",
        dest="exact_match",
        action="store_true",
        help="Match query exactly against item title (and aliases) before returning results.",
    )


def run_main(args: argparse.Namespace) -> None:
    """Run the `reaxkit help` command."""
    if getattr(args, "query", None) is None:
        print("ReaxKit help\n")
        print('Usage:\n  reaxkit help "msd"\n  reaxkit help "bond order"\n  reaxkit help "restraint"')
        print('  reaxkit help "restraint" --engine reaxff\n')
        print("Tip: put multi-word queries in quotes.")
        return

    from reaxkit.help.help_index_loader import (
        build_help_relationship_report,
    )

    engine = getattr(args, "engine", None) or "reaxff"
    report = build_help_relationship_report(
        args.query,
        top_k=max(1, int(getattr(args, "top", 8))),
        engine=engine,
        all_info=bool(getattr(args, "all_info", False)),
        exact_match=bool(getattr(args, "exact_match", False)),
    )
    print(report)


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    """`help` is a kind-level command and intentionally has no task subcommands."""
    return
