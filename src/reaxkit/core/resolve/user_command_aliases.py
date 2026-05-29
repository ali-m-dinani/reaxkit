"""User-defined command alias storage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

import yaml


def user_command_aliases_path() -> Path:
    """Return the user-scoped command alias file path."""
    appdata = os.getenv("APPDATA")
    if appdata:
        return Path(appdata) / "reaxkit" / "command_aliases.yaml"
    return Path.home() / ".reaxkit" / "command_aliases.yaml"


def load_user_command_aliases() -> dict[str, tuple[str, ...]]:
    """Load user-defined command aliases from disk."""
    path = user_command_aliases_path()
    if not path.is_file():
        return {}

    with path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}

    raw = doc.get("commands") or {}
    out: dict[str, tuple[str, ...]] = {}
    for command_name, aliases in raw.items():
        if aliases is None:
            out[str(command_name)] = ()
            continue
        if not isinstance(aliases, (list, tuple)):
            aliases = (aliases,)
        out[str(command_name)] = tuple(str(alias) for alias in aliases if str(alias).strip())
    return out


def save_user_command_aliases(aliases: Mapping[str, Sequence[str]]) -> Path:
    """Persist user-defined command aliases to disk."""
    path = user_command_aliases_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "commands": {
            str(command_name): [str(alias) for alias in command_aliases]
            for command_name, command_aliases in sorted(aliases.items())
            if command_aliases
        }
    }
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=True, allow_unicode=False)
    return path


def add_user_command_alias(command_name: str, alias: str) -> Path:
    """Append a user-defined alias to a canonical command."""
    aliases = load_user_command_aliases()
    existing = list(aliases.get(command_name, ()))
    if alias not in existing:
        existing.append(alias)
    aliases[command_name] = tuple(existing)
    return save_user_command_aliases(aliases)
