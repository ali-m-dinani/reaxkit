"""
Resolve canonical command names from tolerant user-facing aliases.

**Usage context**

- Import these helpers from ReaxKit core modules when implementing CLI and workflow logic.
- Reuse the public APIs here to keep behavior consistent across commands and engines.
"""

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
    """
    Public wrapper for command token normalization.
    
    This function is part of the ReaxKit core API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    value : str
        Input parameter used by this function.
    
    Returns
    -----
    str
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.resolve.command_alias_resolver import normalize_command_token
    # Configure required arguments for your case.
    result = normalize_command_token(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return _normalize_command_token(value)


def registered_command_names() -> list[str]:
    """
    Return the currently registered canonical command names.
    
    This function is part of the ReaxKit core API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    None
    
    Returns
    -----
    list[str]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.resolve.command_alias_resolver import registered_command_names
    # Configure required arguments for your case.
    result = registered_command_names(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return sorted(get_registered_commands())


def registered_task_names() -> list[str]:
    """
    Backward-compatible alias for command discovery.
    
    This function is part of the ReaxKit core API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    None
    
    Returns
    -----
    list[str]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.resolve.command_alias_resolver import registered_task_names
    # Configure required arguments for your case.
    result = registered_task_names(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
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
        
    
    Examples
    -----
    ```python
    from reaxkit.core.resolve.command_alias_resolver import build_command_alias_index
    # Configure required arguments for your case.
    result = build_command_alias_index(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
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
        
    
    Examples
    -----
    ```python
    from reaxkit.core.resolve.command_alias_resolver import resolve_command_name
    # Configure required arguments for your case.
    result = resolve_command_name(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
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
    """
    Backward-compatible alias for command alias index construction.
    
    This function is part of the ReaxKit core API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    task_names : Iterable[str]
        Input parameter used by this function.
    aliases : Mapping[str, Sequence[str]] | None, optional
        Input parameter used by this function.
    
    Returns
    -----
    dict[str, str]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.resolve.command_alias_resolver import build_task_alias_index
    # Configure required arguments for your case.
    result = build_task_alias_index(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return build_command_alias_index(task_names, aliases=aliases)


def resolve_task_name(
    value: str,
    task_names: Iterable[str] | None = None,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """
    Backward-compatible alias for command resolution.
    
    This function is part of the ReaxKit core API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    value : str
        Input parameter used by this function.
    task_names : Iterable[str] | None, optional
        Input parameter used by this function.
    aliases : Mapping[str, Sequence[str]] | None, optional
        Input parameter used by this function.
    
    Returns
    -----
    str
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.resolve.command_alias_resolver import resolve_task_name
    # Configure required arguments for your case.
    result = resolve_task_name(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return resolve_command_name(value, task_names=task_names, aliases=aliases)


def is_known_command(
    value: str,
    task_names: Iterable[str] | None = None,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> bool:
    """
    Return ``True`` when a command token resolves successfully.
    
    This function is part of the ReaxKit core API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    value : str
        Input parameter used by this function.
    task_names : Iterable[str] | None, optional
        Input parameter used by this function.
    aliases : Mapping[str, Sequence[str]] | None, optional
        Input parameter used by this function.
    
    Returns
    -----
    bool
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.resolve.command_alias_resolver import is_known_command
    # Configure required arguments for your case.
    result = is_known_command(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
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
    """
    Backward-compatible alias for command resolution checks.
    
    This function is part of the ReaxKit core API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    value : str
        Input parameter used by this function.
    task_names : Iterable[str] | None, optional
        Input parameter used by this function.
    aliases : Mapping[str, Sequence[str]] | None, optional
        Input parameter used by this function.
    
    Returns
    -----
    bool
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.resolve.command_alias_resolver import is_known_task
    # Configure required arguments for your case.
    result = is_known_task(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return is_known_command(value, task_names=task_names, aliases=aliases)
