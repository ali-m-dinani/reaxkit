"""Direct command workflow for managing user-defined command aliases."""

from __future__ import annotations

import argparse

from reaxkit.core.command_alias_resolver import (
    build_command_alias_index,
    normalize_command_token,
    resolve_command_name,
)
from reaxkit.core.command_catalog import get_registered_commands
from reaxkit.core.user_command_aliases import add_user_command_alias, user_command_aliases_path


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    _ = command
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Add a user-defined alias for an existing ReaxKit command.\n"
        "Use this command to create shorter or more memorable names for frequently used commands.\n"
        "The alias is saved in your user alias configuration and becomes available in future CLI runs.\n\n"
        "Examples:\n"
        "  1. Add a short alias for `timeseries`:\n"
        "   reaxkit add-alias timeseries ts\n\n"
        "  2. Add a custom alias for `mean-square-displacement`:\n"
        "   reaxkit add-alias mean-square-displacement msd2\n\n"
        "  3. Add an alias for `charge-table`:\n"
        "   reaxkit add-alias charge-table charges"
    )
    parser.add_argument(
        "target_command",
        help="Canonical command or any existing alias for it. Example: timeseries, which resolves to the command you want to alias.",
    )
    parser.add_argument(
        "alias",
        help="New alias to add for that command. Example: ts, which becomes a shortcut you can use instead of the full command name.",
    )
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    _ = command
    command_names = get_registered_commands().keys()
    canonical = resolve_command_name(args.target_command, task_names=command_names)
    alias = str(args.alias).strip()
    normalized_alias = normalize_command_token(alias)

    if not normalized_alias:
        raise ValueError("Alias cannot be empty.")

    current_alias_index = build_command_alias_index(command_names)
    existing = current_alias_index.get(normalized_alias)
    if existing is not None and existing != canonical:
        raise ValueError(
            f"Alias {alias!r} is already assigned to command {existing!r}. "
            "Choose a different alias."
        )

    out_path = add_user_command_alias(canonical, alias)
    print(f"[Done] Added alias {alias!r} for command {canonical!r}.")
    print(f"[Done] Saved user aliases to {out_path}")
    return 0
