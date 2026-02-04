"""
Interactive help and discovery workflow for ReaxKit.

This workflow provides a search-based help system that maps conceptual,
human-language queries (e.g. "electric field", "restraint", "bond order")
to relevant ReaxFF input and output files.

It operates as a kind-level command (`reaxkit help`) rather than a task-based
workflow, and therefore does not define subcommands. Instead, it queries
curated help indices and ranks matches based on relevance.

Optional flags allow users to inspect why a file matched, view example
ReaxKit commands, and explore core, optional, and derived variables
associated with each file.
"""

from __future__ import annotations

import argparse
from reaxkit.help.help_index_loader import search_help_indices, _format_hits


def build_parser(p: argparse.ArgumentParser) -> None:
    """
    Define CLI arguments for the kind-level command: `reaxkit help`.

    This is used by documentation generation to render a stable list of flags
    and defaults, and can also be reused by the CLI entry-point when building
    the parser for `help`.
    """
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
        help="Maximum number of matches to display.",
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=35.0,
        help="Minimum score threshold; lower-scoring matches are hidden.",
    )

    # Output detail toggles
    p.add_argument("--why", action="store_true", help="Show why each file matched the query.")
    p.add_argument("--file_templates", action="store_true", help="Show one example ReaxKit command per match.")
    p.add_argument("--tags", action="store_true", help="Show tags for each match.")
    p.add_argument("--core-vars", dest="core_vars", action="store_true", help="Show core variables for each match.")
    p.add_argument("--optional-vars", dest="optional_vars", action="store_true", help="Show optional variables.")
    p.add_argument("--derived-vars", dest="derived_vars", action="store_true", help="Show derived variables.")
    p.add_argument("--notes", action="store_true", help="Show notes for each match.")

    # Convenience flag
    p.add_argument(
        "--all-info",
        dest="all_info",
        action="store_true",
        help="Show why/file_templates/tags/core/optional/derived/notes all at once.",
    )


def run_main(args: argparse.Namespace) -> None:
    """
    Run the `reaxkit help` command.

    Behavior:
    - If no query is provided, prints a short usage message and exits.
    - If a query is provided, searches curated help indices and prints ranked matches.
    - Use `--top` and `--min-score` to control result count and filtering.
    - Use detail flags (`--why`, `--file_templates`, `--tags`, `--core-vars`, `--optional-vars`,
      `--derived-vars`, `--notes`) to expand what is shown per match.
    - `--all-info` enables all detail flags together.
    """
    # Allow: `reaxkit help` (no query)
    if getattr(args, "query", None) is None:
        print("ReaxKit help\n")
        print('Usage:\n  reaxkit help "restraint"\n  reaxkit help bond\n  reaxkit help "electric field"\n')
        print("Tip: put multi-word queries in quotes.")
        return

    hits = search_help_indices(
        args.query,
        top_k=getattr(args, "top", 8),
        min_score=getattr(args, "min_score", 35.0),
    )

    if not hits:
        print(f"❌ No matches for: {args.query!r}")
        return

    all_info = getattr(args, "all_info", False)

    show_why = all_info or getattr(args, "why", False)
    show_examples = all_info or getattr(args, "file_templates", False)
    show_tags = all_info or getattr(args, "tags", False)
    show_core_vars = all_info or getattr(args, "core_vars", False)
    show_optional_vars = all_info or getattr(args, "optional_vars", False)
    show_derived_vars = all_info or getattr(args, "derived_vars", False)
    show_notes = all_info or getattr(args, "notes", False)

    print(
        _format_hits(
            hits,
            show_why=show_why,
            show_examples=show_examples,
            show_tags=show_tags,
            show_core_vars=show_core_vars,
            show_optional_vars=show_optional_vars,
            show_derived_vars=show_derived_vars,
            show_notes=show_notes,
        )
    )


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    """
    Intentionally empty.

    `help` is a kind-level command (invoked as `reaxkit help "<query>"`), so it does
    not define subcommands via `register_tasks`. The CLI entry-point should call
    `build_parser(...)` and route execution to `run_main(...)`.
    """
    return
