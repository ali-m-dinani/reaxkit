"""Naming helpers for study entities."""

from __future__ import annotations

from typing import Any


def slug(value: Any) -> str:
    text = str(value).strip()
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    norm = "".join(out).strip("_")
    return norm or "value"


def slug_underscore(value: Any) -> str:
    return slug(value).replace("-", "_").replace(".", "_")


def canonical_token(text: str) -> str:
    return "".join(ch.lower() for ch in str(text) if ch.isalnum())


def short_param_name(name: str) -> str:
    low = str(name).strip().lower()
    if low.endswith("_percent"):
        return low[: -len("_percent")]
    if low == "temperature":
        return "temp"
    return low


def compact_param_name(name: str) -> str:
    low = str(name).strip().lower()
    if not low:
        return "p"
    token = low.split("_", 1)[0]
    token = token[:2] if len(token) >= 2 else token
    return token or "p"


def format_param_value_for_case(value: Any) -> str:
    try:
        f = float(value)
        if f.is_integer():
            i = int(f)
            return f"{i:02d}" if 0 <= i < 100 else str(i)
    except Exception:
        pass
    return slug(value)


def case_label_from_params(params: dict[str, Any]) -> str:
    parts = [f"{short_param_name(k)}_{format_param_value_for_case(v)}" for k, v in params.items()]
    return "__".join(parts)


def case_label_from_params_compact(params: dict[str, Any]) -> str:
    parts = [f"{compact_param_name(k)}_{format_param_value_for_case(v)}" for k, v in params.items()]
    return "_".join(parts)

