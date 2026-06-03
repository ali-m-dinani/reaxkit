"""
User-defined command alias storage.

**Usage context**

- Import these helpers from ReaxKit core modules when implementing CLI and workflow logic.
- Reuse the public APIs here to keep behavior consistent across commands and engines.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

import yaml


def user_command_aliases_path() -> Path:
    """
    Return the user-scoped command alias file path.
    
    This function is part of the ReaxKit core API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    None
    
    Returns
    -----
    Path
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.resolve.user_command_aliases import user_command_aliases_path
    # Configure required arguments for your case.
    result = user_command_aliases_path(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    appdata = os.getenv("APPDATA")
    if appdata:
        return Path(appdata) / "reaxkit" / "command_aliases.yaml"
    return Path.home() / ".reaxkit" / "command_aliases.yaml"


def load_user_command_aliases() -> dict[str, tuple[str, ...]]:
    """
    Load user-defined command aliases from disk.
    
    This function is part of the ReaxKit core API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    None
    
    Returns
    -----
    dict[str, tuple[str, ...]]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.resolve.user_command_aliases import load_user_command_aliases
    # Configure required arguments for your case.
    result = load_user_command_aliases(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
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
    """
    Persist user-defined command aliases to disk.
    
    This function is part of the ReaxKit core API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    aliases : Mapping[str, Sequence[str]]
        Input parameter used by this function.
    
    Returns
    -----
    Path
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.resolve.user_command_aliases import save_user_command_aliases
    # Configure required arguments for your case.
    result = save_user_command_aliases(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
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
    """
    Append a user-defined alias to a canonical command.
    
    This function is part of the ReaxKit core API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    command_name : str
        Input parameter used by this function.
    alias : str
        Input parameter used by this function.
    
    Returns
    -----
    Path
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.resolve.user_command_aliases import add_user_command_alias
    # Configure required arguments for your case.
    result = add_user_command_alias(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    aliases = load_user_command_aliases()
    existing = list(aliases.get(command_name, ()))
    if alias not in existing:
        existing.append(alias)
    aliases[command_name] = tuple(existing)
    return save_user_command_aliases(aliases)
