from __future__ import annotations

import argparse
from reaxkit.help.help_index_loader import search_help_indices, format_hits


def run_main(args: argparse.Namespace) -> None:
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
    show_examples = all_info or getattr(args, "examples", False)
    show_tags = all_info or getattr(args, "tags", False)
    show_core_vars = all_info or getattr(args, "core_vars", False)
    show_optional_vars = all_info or getattr(args, "optional_vars", False)
    show_derived_vars = all_info or getattr(args, "derived_vars", False)
    show_notes = all_info or getattr(args, "notes", False)

    print(
        format_hits(
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
    No subcommands are registered for `reaxkit help`.

    `help` is a kind-level command (like `intspec`) and is invoked directly as:
        reaxkit help "<query>"

    It searches curated help indices to map a conceptual query (e.g. "electric field")
    to relevant ReaxFF input/output files.

    Supported options:
        --top        Limit the number of displayed matches.
        --min-score  Filter out weak matches.

        --why              Show brief reasons for why each file matched the query.
        --examples         Show one example ReaxKit command per matched file.
        --tags             Show tags for each matched file.
        --core_vars        Show core variables for each matched file.
        --optional-vars    Show optional variables for each matched file.
        --derived_vars     Show derived variables for each matched file.
        --notes            Show notes for each matched file.
        --all-info         Show why/examples/tags/core/optional/derived/notes.


    This function is intentionally empty because all logic is handled by the
    kind-level runner (`run_main`).
    """
    return

