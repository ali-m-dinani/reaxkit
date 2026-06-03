"""Interactive help and discovery workflow for ReaxKit.

This workflow supports two complementary help modes:
1) capability discovery (intent -> responsible analysis tasks/workflows/generators)
2) ReaxFF file discovery (query -> relevant input/output files)

**Usage context**

- Command routing: Resolve CLI aliases and normalized command names.
- Task execution: Build request objects and invoke registered tasks.
- Output handling: Forward results to table, plot, export, or report flows.
"""

from __future__ import annotations

import argparse

ALL_COMMANDS = ("help",)
ALL_LEGACY_COMMANDS = ()


def build_parser(p: argparse.ArgumentParser) -> None:
    """Define CLI arguments for `reaxkit help`."""
    p.formatter_class = argparse.RawTextHelpFormatter
    p.description = (
        "Interactive help and discovery for ReaxKit commands, capabilities, and file semantics.\n"
        "Use this command to search ReaxKit concepts (for example analyses or generators) and\n"
        "ReaxFF-related files by keyword. You can narrow results, enforce exact matching, and\n"
        "request detailed mapping information.\n\n"
        
        "For more information, you can see:\n"
        " ReaxKit code: https://github.com/ali-m-dinani/reaxkit\n"
        " ReaxFF documentation: https://ali-m-dinani.github.io/reaxkit/\n\n"
        
        "Examples:\n"
        "  1. Basic keyword search:\n"
        "   reaxkit help \"msd\"\n\n"
        "  2. Search with multi-word phrase:\n"
        "   reaxkit help \"bond order\"\n\n"
        "  3. Limit result count:\n"
        "   reaxkit help \"bond order\" --top 3\n\n"
        "  4. Search with explicit engine context:\n"
        "   reaxkit help \"restraint\" --engine reaxff\n\n"
        "  5. Show detailed mapping information:\n"
        "   reaxkit help \"fort.7\" --all-info\n"
        "   reaxkit help \"xmolout\" --all-info"
    )

    p.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Search query (use quotes for multi-word queries). Example: \"bond order\", which searches that phrase across help index entries.",
    )

    p.add_argument(
        "--top",
        type=int,
        default=1,
        help="Maximum results per category (generator/file/analyzer/workflow), sorted by score. Example: --top 3, which returns only the top 3 hits per category.",
    )

    p.add_argument(
        "--engine",
        type=str,
        default=None,
        help="Optional engine context for dataclass-to-file mappings. Example: --engine reaxff, which resolves relationships using ReaxFF context.",
    )

    p.add_argument(
        "--all-info",
        dest="all_info",
        action="store_true",
        help="Show detailed implementation and file/dataclass/analyzer mapping information. Example: --all-info, which expands output beyond summary hits.",
    )

    p.add_argument(
        "--exact-match",
        dest="exact_match",
        action="store_true",
        help="Match query exactly against item title (and aliases) before returning results. Example: --exact-match, which avoids broad fuzzy matches.",
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
