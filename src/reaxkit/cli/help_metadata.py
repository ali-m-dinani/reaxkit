"""Purpose and visibility of public CLI options.

The canonical option is the first long spelling on an argparse action. Shared
flags are registered here once; individual parsers may set ``help_metadata``
on an action to override its role without changing parsing behavior.
"""

from __future__ import annotations

import argparse

from reaxkit.cli.help_flag_registry import FLAG_METADATA

CATEGORIES = (
    "Scientific choices",
    "Input and file selection",
    "Outputs and plots",
    "Execution",
    "Storage and cache",
    "Diagnostics and compatibility",
)


# Parser paths have the same spelling in terminal help and generated docs.
# These flags describe management/output selection rather than scientific data.
PARSER_OVERRIDES = {
    "study": {
        "--root": ("Outputs and plots", "short"),
        "--analysis": ("Scientific choices", "short"),
        "--action": ("Storage and cache", "full"),
        "--target": ("Storage and cache", "full"),
        "--run": ("Execution", "full"),
        "--stage": ("Execution", "full"),
        "--run-geometry-generator": ("Execution", "full"),
    },
    "manage-workspace": {"--action": ("Storage and cache", "short")},
    "free-up": {"--action": ("Storage and cache", "short")},
    "help": {
        "--top": ("Outputs and plots", "short"),
        "--exact-match": ("Input and file selection", "short"),
    },
    "gen-video": {"--ext": ("Input and file selection", "full")},
    "fort7 get": {"--yaxis": ("Scientific choices", "short")},
}


def metadata_for_flag(canonical: str, parser_path: str = "") -> tuple[str, str]:
    path = parser_path.removeprefix("reaxkit CLI ").removeprefix("reaxkit ")
    override = PARSER_OVERRIDES.get(path, {}).get(canonical)
    if override is not None:
        return override
    if not canonical.startswith("-"):
        return "Input and file selection", "short"
    try:
        return FLAG_METADATA[canonical]
    except KeyError as exc:
        raise ValueError(f"Uncategorized CLI option {canonical} on {parser_path}") from exc


def metadata_for_action(action: argparse.Action, parser: argparse.ArgumentParser) -> tuple[str, str]:
    """Return the explicit role of an action, promoting required inputs."""
    override = getattr(action, "help_metadata", None)
    if override is not None:
        category, visibility = override
    else:
        canonical = next((s for s in action.option_strings if s.startswith("--")),
                         action.option_strings[0] if action.option_strings else action.dest)
        category, visibility = metadata_for_flag(canonical, parser.prog)
        # A modern --input path is the ordinary source selector. Commands that
        # only expose a named source file still show it in short help.
        if canonical in {"--xmolout", "--fort7", "--run-dir"} and "--input" in parser._option_string_actions:
            visibility = "full"
    if category not in CATEGORIES or visibility not in {"short", "full", "internal"}:
        raise ValueError(f"Invalid help metadata for {action.option_strings}: {(category, visibility)}")
    if visibility == "internal" and action.help != argparse.SUPPRESS:
        raise ValueError(f"Visible action cannot be internal: {action.option_strings}")
    if action.required:
        visibility = "short"
    return category, visibility

