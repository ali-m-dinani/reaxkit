"""Backward-compatible imports for column alias resolution."""

from reaxkit.core.resolve.alias import (
    _resolve_alias,
    available_keys as _available_keys,
    load_default_alias_map,
    normalize_choice as _normalize_choice,
    resolve_alias_from_columns,
)


def available_keys(cols):
    """Return current canonical keys plus legacy short canonical names."""
    keys = set(_available_keys(cols))
    if "iterations" in keys:
        keys.add("iter")
    if "potential_energy" in keys:
        keys.add("E_pot")
    return sorted(keys)


def normalize_choice(value: str, domain: str = "xaxis") -> str:
    """Normalize aliases while preserving legacy short canonical choices."""
    token = (value or "").strip().lower()
    if token in {"iter", "e_pot"}:
        return "iter" if token == "iter" else "E_pot"
    return _normalize_choice(value, domain=domain)

__all__ = [
    "_resolve_alias",
    "available_keys",
    "load_default_alias_map",
    "normalize_choice",
    "resolve_alias_from_columns",
]
