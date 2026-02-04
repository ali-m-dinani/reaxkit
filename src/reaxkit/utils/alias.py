"""
Alias resolution utilities for tolerant column and key matching.

This module provides functions for resolving canonical ReaxKit keys
(e.g., ``iter``, ``time``, ``D``) against the actual column names present
in parsed DataFrames, using a packaged alias map.

The canonical→alias definitions are stored in ``reaxkit/data/aliases.yaml``.
"""

from __future__ import annotations

from typing import Dict, List, Iterable, Optional
from functools import lru_cache

# You can load this via importlib.resources so it works after pip install.
# Requires: aliases.yaml included as package data.
import yaml
import importlib.resources as ir


@lru_cache(maxsize=1)
def load_default_alias_map() -> Dict[str, List[str]]:
    """
    Load the packaged canonical→aliases mapping.

    The alias map is read from ``reaxkit/data/aliases.yaml`` and cached after
    the first call.

    Returns
    -------
    dict[str, list[str]]
        Mapping of canonical keys to accepted alias strings.

    Raises
    ------
    FileNotFoundError
        If the packaged ``aliases.yaml`` cannot be found.
    """
    # reaxkit.data is NOT a package; we read by file location within package resources.
    # If you later make data/ a package, you can switch to ir.files("reaxkit.data").
    pkg = "reaxkit"
    rel = "data/aliases.yaml"

    try:
        with ir.files(pkg).joinpath(rel).open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find packaged alias map at '{pkg}/{rel}'. "
            "Make sure aliases.yaml is included as package data."
        ) from e

    aliases = doc.get("aliases") or {}
    # Normalize to Dict[str, List[str]]
    out: Dict[str, List[str]] = {}
    for k, v in aliases.items():
        if v is None:
            out[str(k)] = []
        elif isinstance(v, list):
            out[str(k)] = [str(x) for x in v]
        else:
            out[str(k)] = [str(v)]
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
        Available column names (e.g., DataFrame columns).
    canonical : str
        Canonical key to resolve (e.g., ``"iter"``, ``"time"``, ``"D"``).
    aliases : dict[str, list[str]], optional
        Canonical→aliases mapping to use. If not provided, the packaged map
        from ``aliases.yaml`` is loaded.

    Returns
    -------
    str or None
        The matching column name if found, otherwise ``None``.

    Examples
    --------
    >>> resolve_alias_from_columns(df.columns, "time")
    """
    if cols is None:
        return None

    orig_cols = list(cols)
    lower_map = {c.lower(): c for c in orig_cols}
    aliases = aliases or load_default_alias_map()

    candidates = [canonical]
    if canonical in aliases:
        candidates.extend(aliases[canonical])

    # Exact (case-insensitive)
    for cand in candidates:
        hit = lower_map.get(str(cand).lower())
        if hit is not None:
            return hit

    # Heuristics on canonical (startswith/contains)
    cname = str(canonical).lower()
    for c in orig_cols:
        cl = c.lower()
        if cl.startswith(cname) or cname in cl:
            return c

    return None


def _resolve_alias(source, canonical: str) -> str:
    """
    Resolve a canonical key from a DataFrame-like source.

    Notes
    -----
    This is a compatibility helper that accepts a handler (with ``.dataframe()``),
    a pandas DataFrame, or an iterable of column names.
    """
    try:
        cols = list(source.dataframe().columns)  # type: ignore[attr-defined]
    except Exception:
        try:
            cols = list(getattr(source, "columns"))
        except Exception:
            cols = list(source)  # assume iterable of str

    hit = resolve_alias_from_columns(cols, canonical)
    if hit is None:
        raise KeyError(
            f"Could not resolve alias '{canonical}'. Available columns: {list(cols)}"
        )
    return hit


def _available_keys_from_columns(cols: Iterable[str]) -> List[str]:
    """
    List canonical keys that are usable for a given column set.

    The returned list includes:
    - raw columns already present in ``cols``
    - canonical keys whose aliases resolve against ``cols``

    Parameters
    ----------
    cols : iterable of str
        Available column names.

    Returns
    -------
    list[str]
        Sorted list of usable keys for lookup and CLI choices.

    Examples
    --------
    >>> _available_keys_from_columns(df.columns)
    """
    amap = load_default_alias_map()
    cols_set = set(cols)
    keys = set(cols_set)
    for alias, cands in amap.items():
        if any(c in cols_set for c in cands) or alias in cols_set:
            keys.add(alias)
    return sorted(keys)


# Re-export for callers that already import these names
available_keys = _available_keys_from_columns


def normalize_choice(value: str, domain: str = "xaxis") -> str:
    """
    Normalize a user-provided keyword to its canonical alias key.

    This is intended for tolerant CLI inputs where users may provide
    any alias defined in ``aliases.yaml`` (e.g., ``Time(fs)`` → ``time``).

    Parameters
    ----------
    value : str
        User-provided keyword or alias.
    domain : str, optional
        Reserved for future domain-specific normalization rules.

    Returns
    -------
    str
        Canonical key if an alias match is found; otherwise the normalized
        input string.

    Examples
    --------
    >>> normalize_choice("Time(fs)")
    >>> normalize_choice("frm")
    """
    v = (value or "").strip().lower()
    if not v:
        return v

    amap = load_default_alias_map()
    for canonical, aliases in amap.items():
        all_names = [canonical.lower()] + [a.lower() for a in aliases]
        if v in all_names:
            return canonical

    return v
