"""
Alias resolution utilities for tolerant variable and column matching.

This module provides functions for resolving canonical ReaxKit keys
(e.g., ``iterations``, ``time``, ``density``) against the actual column
names present in parsed DataFrames, using a packaged alias map.

The canonical-to-alias definitions are stored in
``reaxkit/data/variable_aliases.yaml``.

**Usage context**

- Import these helpers from ReaxKit core modules when implementing CLI and workflow logic.
- Reuse the public APIs here to keep behavior consistent across commands and engines.
"""

from __future__ import annotations

from functools import lru_cache
import importlib.resources as ir
from typing import Dict, Iterable, List, Optional

import yaml


@lru_cache(maxsize=1)
def load_default_alias_map() -> Dict[str, List[str]]:
    """
    Load the packaged canonical-to-aliases mapping.
    
        The alias map is read from ``reaxkit/data/variable_aliases.yaml`` and cached
        after the first call.
    
        Returns
        -------
        dict[str, list[str]]
            Mapping of canonical keys to accepted alias strings.
    
        Raises
        ------
        FileNotFoundError
            If the packaged ``variable_aliases.yaml`` cannot be found.
        
    
    Parameters
    -----
    None
    
    Examples
    -----
    ```python
    from reaxkit.core.resolve.alias import load_default_alias_map
    # Configure required arguments for your case.
    result = load_default_alias_map(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    pkg = "reaxkit"
    rel = "data/variable_aliases.yaml"

    try:
        with ir.files(pkg).joinpath(rel).open("r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find packaged alias map at '{pkg}/{rel}'. "
            "Make sure variable_aliases.yaml is included as package data."
        ) from e

    aliases = doc.get("variable_aliases") or {}
    out: Dict[str, List[str]] = {}
    for key, value in aliases.items():
        if value is None:
            out[str(key)] = []
        elif isinstance(value, list):
            out[str(key)] = [str(item) for item in value]
        else:
            out[str(key)] = [str(value)]
    return out


def resolve_alias_from_columns(
    cols: Iterable[str],
    canonical: str,
    aliases: Optional[Dict[str, List[str]]] = None,
) -> Optional[str]:
    """
    Resolve a canonical key to the matching column name in a column list.
    
        Matching is case-insensitive and falls back to simple heuristics when an
        exact alias match is not found.
    
        Parameters
        ----------
        cols : iterable of str
            Available column names.
        canonical : str
            Canonical key to resolve (e.g., ``"iterations"``, ``"time"``,
            ``"density"``).
        aliases : dict[str, list[str]], optional
            Canonical-to-aliases mapping to use. If not provided, the packaged map
            from ``variable_aliases.yaml`` is loaded.
    
        Returns
        -------
        str or None
            The matching column name if found, otherwise ``None``.
        
    
    Examples
    -----
    ```python
    from reaxkit.core.resolve.alias import resolve_alias_from_columns
    # Configure required arguments for your case.
    result = resolve_alias_from_columns(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    if cols is None:
        return None

    orig_cols = list(cols)
    lower_map = {c.lower(): c for c in orig_cols}
    aliases = aliases or load_default_alias_map()

    candidates = [canonical]
    if canonical in aliases:
        candidates.extend(aliases[canonical])

    for cand in candidates:
        hit = lower_map.get(str(cand).lower())
        if hit is not None:
            return hit

    canonical_lower = str(canonical).lower()
    for col in orig_cols:
        col_lower = col.lower()
        if col_lower.startswith(canonical_lower) or canonical_lower in col_lower:
            return col

    return None


def _resolve_alias(source, canonical: str) -> str:
    """
    Resolve a canonical key from a DataFrame-like source.

    This compatibility helper accepts a handler (with ``.dataframe()``),
    a pandas DataFrame, or an iterable of column names.
    """
    try:
        cols = list(source.dataframe().columns)  # type: ignore[attr-defined]
    except Exception:
        try:
            cols = list(getattr(source, "columns"))
        except Exception:
            cols = list(source)

    hit = resolve_alias_from_columns(cols, canonical)
    if hit is None:
        raise KeyError(f"Could not resolve alias '{canonical}'. Available columns: {list(cols)}")
    return hit


def _available_keys_from_columns(cols: Iterable[str]) -> List[str]:
    """
    List canonical keys that are usable for a given column set.

    The returned list includes:
    - raw columns already present in ``cols``
    - canonical keys whose aliases resolve against ``cols``
    """
    alias_map = load_default_alias_map()
    cols_set = set(cols)
    keys = set(cols_set)
    for alias, candidates in alias_map.items():
        if any(candidate in cols_set for candidate in candidates) or alias in cols_set:
            keys.add(alias)
    return sorted(keys)


available_keys = _available_keys_from_columns


def normalize_choice(value: str, domain: str = "xaxis") -> str:
    """
    Normalize a user-provided keyword to its canonical alias key.
    
        This is intended for tolerant CLI inputs where users may provide any alias
        defined in ``variable_aliases.yaml`` (e.g., ``Time(fs)`` -> ``time``).
        
    
    Parameters
    -----
    value : str
        Input parameter used by this function.
    domain : str, optional
        Input parameter used by this function.
    
    Returns
    -----
    str
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.resolve.alias import normalize_choice
    # Configure required arguments for your case.
    result = normalize_choice(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    _ = domain
    normalized_value = (value or "").strip().lower()
    if not normalized_value:
        return normalized_value

    alias_map = load_default_alias_map()
    for canonical, aliases in alias_map.items():
        all_names = [canonical.lower()] + [alias.lower() for alias in aliases]
        if normalized_value in all_names:
            return canonical

    return normalized_value
