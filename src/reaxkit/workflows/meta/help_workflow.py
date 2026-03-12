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
        "  reaxkit help \"bond order\" --top 12\n"
        "  reaxkit help \"restraint\" --engine reaxff\n"
        "  reaxkit help \"fort.7\" --why --tags\n"
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
        help="Maximum number of matches to display (for capability and file matches).",
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=35.0,
        help="Minimum score threshold; lower-scoring matches are hidden.",
    )

    p.add_argument(
        "--engine",
        type=str,
        default=None,
        help="Optional engine context (example: reaxff) for dataclass-to-file mappings.",
    )

    # Output detail toggles for file-index results
    p.add_argument("--why", action="store_true", help="Show why each file matched the query.")
    p.add_argument("--file_templates", action="store_true", help="Show one example command per file match.")
    p.add_argument("--tags", action="store_true", help="Show tags for each file match.")
    p.add_argument("--core-vars", dest="core_vars", action="store_true", help="Show core variables.")
    p.add_argument("--optional-vars", dest="optional_vars", action="store_true", help="Show optional variables.")
    p.add_argument("--derived-vars", dest="derived_vars", action="store_true", help="Show derived variables.")
    p.add_argument("--notes", action="store_true", help="Show notes for each file match.")

    p.add_argument(
        "--all-info",
        dest="all_info",
        action="store_true",
        help="Show why/file_templates/tags/core/optional/derived/notes all at once.",
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
        _format_command_hits,
        _format_hits,
        search_help_indices,
        search_help_commands,
    )

    command_hits = search_help_commands(
        args.query,
        top_k=getattr(args, "top", 8),
        min_score=getattr(args, "min_score", 35.0),
    )
    file_hits = search_help_indices(
        args.query,
        top_k=getattr(args, "top", 8),
        min_score=getattr(args, "min_score", 35.0),
    )

    if not command_hits and not file_hits:
        print(f"No matches for: {args.query!r}")
        return

    all_info = getattr(args, "all_info", False)
    show_why = all_info or getattr(args, "why", False)
    show_examples = all_info or getattr(args, "file_templates", False)
    show_tags = all_info or getattr(args, "tags", False)
    show_core_vars = all_info or getattr(args, "core_vars", False)
    show_optional_vars = all_info or getattr(args, "optional_vars", False)
    show_derived_vars = all_info or getattr(args, "derived_vars", False)
    show_notes = all_info or getattr(args, "notes", False)

    engine = getattr(args, "engine", None) or "reaxff"

    if command_hits:
        print(_format_command_hits(command_hits))
        if file_hits:
            print("\n-------------")

    if file_hits:
        print(
            _format_hits(
                file_hits,
                show_why=show_why,
                show_examples=show_examples,
                show_tags=show_tags,
                show_core_vars=show_core_vars,
                show_optional_vars=show_optional_vars,
                show_derived_vars=show_derived_vars,
                show_notes=show_notes,
                engine=engine,
            )
        )


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    """`help` is a kind-level command and intentionally has no task subcommands."""
    return
