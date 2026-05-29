"""Resolve canonical command names from tolerant user-facing aliases."""

from __future__ import annotations

from difflib import get_close_matches
import re
from typing import Iterable, Mapping, Sequence

from reaxkit.core.registry.command_catalog import get_registered_commands
from reaxkit.core.resolve.user_command_aliases import load_user_command_aliases


def _normalize_command_token(value: str) -> str:
    """Normalize a command token for case-insensitive alias matching."""
    token = (value or "").strip().lower()
    token = re.sub(r"[\s\-_]+", "", token)
    return token


def normalize_command_token(value: str) -> str:
    """Public wrapper for command token normalization."""
    return _normalize_command_token(value)


def registered_command_names() -> list[str]:
    """Return the currently registered canonical command names."""
    return sorted(get_registered_commands())


def registered_task_names() -> list[str]:
    """Backward-compatible alias for command discovery."""
    return registered_command_names()


def build_command_alias_index(
    command_names: Iterable[str],
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, str]:
    """
    Build a normalized alias index for command resolution.

    Parameters
    ----------
    command_names
        Canonical command names.
    aliases
        Optional explicit aliases keyed by canonical command name.

    Returns
    -------
    dict[str, str]
        Mapping from normalized alias token to canonical command name.
    """
    alias_index: dict[str, str] = {}
    explicit_aliases = aliases or {}
    registry_aliases = {
        name: spec.aliases
        for name, spec in get_registered_commands().items()
    }
    user_aliases = load_user_command_aliases()

    for command in command_names:
        canonical = str(command)
        candidates = [canonical, canonical.replace("_", "-"), canonical.replace("-", "_")]
        candidates.extend(registry_aliases.get(canonical, ()))
        candidates.extend(user_aliases.get(canonical, ()))
        candidates.extend(explicit_aliases.get(canonical, ()))

        for candidate in candidates:
            normalized = _normalize_command_token(candidate)
            if normalized:
                alias_index[normalized] = canonical

    return alias_index


def resolve_command_name(
    value: str,
    task_names: Iterable[str] | None = None,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """
    Resolve a user-provided command token to a canonical command name.

    Parameters
    ----------
    value
        User-supplied command token such as ``"msd"`` or
        ``"mean-square-displacement"``.
    task_names
        Known canonical command names. Defaults to the registered commands.
    aliases
        Optional explicit aliases keyed by canonical command name.

    Returns
    -------
    str
        Canonical command name.

    Raises
    ------
    KeyError
        If the command cannot be resolved.
    """
    names = list(task_names) if task_names is not None else registered_command_names()
    if not names:
        raise KeyError("No commands are registered for alias resolution.")

    normalized = _normalize_command_token(value)
    if not normalized:
        raise KeyError("Command name cannot be empty.")

    alias_index = build_command_alias_index(names, aliases=aliases)
    resolved = alias_index.get(normalized)
    if resolved is not None:
        return resolved

    suggestions = get_close_matches(normalized, alias_index.keys(), n=3, cutoff=0.6)
    canonical_suggestions: list[str] = []
    for item in suggestions:
        canonical = alias_index[item]
        if canonical not in canonical_suggestions:
            canonical_suggestions.append(canonical)

    message = f"Unknown command alias '{value}'."
    if canonical_suggestions:
        message += f" Did you mean: {', '.join(canonical_suggestions)}?"
    raise KeyError(message)


def build_task_alias_index(
    task_names: Iterable[str],
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, str]:
    """Backward-compatible alias for command alias index construction."""
    return build_command_alias_index(task_names, aliases=aliases)


def resolve_task_name(
    value: str,
    task_names: Iterable[str] | None = None,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """Backward-compatible alias for command resolution."""
    return resolve_command_name(value, task_names=task_names, aliases=aliases)


def is_known_command(
    value: str,
    task_names: Iterable[str] | None = None,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> bool:
    """Return ``True`` when a command token resolves successfully."""
    try:
        resolve_command_name(value, task_names=task_names, aliases=aliases)
    except KeyError:
        return False
    return True


def is_known_task(
    value: str,
    task_names: Iterable[str] | None = None,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> bool:
    """Backward-compatible alias for command resolution checks."""
    return is_known_command(value, task_names=task_names, aliases=aliases)
