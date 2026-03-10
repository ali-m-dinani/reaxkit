"""Small helpers for task-specific request forms."""

from __future__ import annotations

from typing import Any

from dash import html


def schema_field(schema: dict[str, Any] | None, field_name: str) -> dict[str, Any]:
    fields = schema.get("fields", []) if isinstance(schema, dict) else []
    for field in fields:
        if isinstance(field, dict) and str(field.get("name")) == str(field_name):
            return field
    return {}


def semantic_value(schema: dict[str, Any] | None, field_name: str, key: str, default: Any = None) -> Any:
    field = schema_field(schema, field_name)
    semantic = field.get("semantic", {}) if isinstance(field, dict) else {}
    if isinstance(semantic, dict) and key in semantic:
        return semantic.get(key)
    return default


def semantic_choices(schema: dict[str, Any] | None, field_name: str, default: list[Any] | None = None) -> list[Any]:
    raw = semantic_value(schema, field_name, "choices", default or [])
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    return list(default or [])


def label_with_help(label: str, help_text: str | None, *, tooltip_style: str = "dot") -> Any:
    text = str(help_text or "").strip()
    if not text:
        return html.Label(label)
    if tooltip_style == "inline":
        return html.Div([html.Span(label), html.Small(text)], className="rk-help-inline")
    return html.Div([html.Span(label), html.Span("?", className="rk-help-dot", title=text)], className="rk-help-inline")


__all__ = [
    "label_with_help",
    "schema_field",
    "semantic_choices",
    "semantic_value",
]
