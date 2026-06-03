"""Task-specific analysis callback hooks."""

from __future__ import annotations

from typing import Any

from reaxkit.webui.backend.api import WebUIApiService
from reaxkit.webui.ui.analysis.tasks.autogen import (
    register_auto_task_callbacks,
    render_auto_task_form,
)


def register_task_callbacks(
    app,
    service: WebUIApiService,
    *,
    selected_node,
    parse_csv_ints,  # kept for compatibility
    parse_csv_strs,  # kept for compatibility
) -> None:
    _ = parse_csv_ints
    _ = parse_csv_strs
    register_auto_task_callbacks(
        app,
        service,
        selected_node=selected_node,
    )


def render_task_analysis_properties(
    node: dict[str, Any],
    lines: list[Any],
    *,
    schema: dict[str, Any] | None = None,
) -> Any | None:
    task_name = str(node.get("metadata", {}).get("task_name") or node.get("name") or "").lower()
    if not task_name or not isinstance(schema, dict):
        return None
    return render_auto_task_form(lines, node, task_name=task_name, schema=schema)


def hidden_task_inputs() -> list[Any]:
    # Auto-generated task forms no longer depend on hardcoded hidden inputs.
    return []


__all__ = [
    "hidden_task_inputs",
    "register_task_callbacks",
    "render_task_analysis_properties",
]
