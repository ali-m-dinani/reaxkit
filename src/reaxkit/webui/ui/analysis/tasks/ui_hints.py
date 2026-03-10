"""Web UI rendering hints for task request fields.

These hints are presentation-only and intentionally kept out of analysis request dataclasses.
"""

from __future__ import annotations

from typing import Any


TASK_UI_HINTS: dict[str, dict[str, Any]] = {
    "msd": {
        "fields": {
            "atom_ids": {"widget": "text_csv_int", "group": "Selection", "tooltip_style": "dot", "layout": "full"},
            "atom_types": {"widget": "text_csv_str", "group": "Selection", "tooltip_style": "dot", "layout": "full"},
            "dims": {"widget": "dropdown", "group": "Geometry", "tooltip_style": "dot", "layout": "inline"},
            "origin": {"widget": "text", "group": "Geometry", "tooltip_style": "dot", "layout": "half"},
            "frames": {"widget": "text_csv_int", "group": "Sampling", "tooltip_style": "dot", "layout": "full"},
            "every": {"widget": "number", "group": "Sampling", "tooltip_style": "dot", "layout": "half"},
        }
    },
    "rdf": {
        "fields": {
            "atom_ids_a": {"widget": "text_csv_int", "group": "Group A", "tooltip_style": "dot", "layout": "full"},
            "atom_types_a": {"widget": "text_csv_str", "group": "Group A", "tooltip_style": "dot", "layout": "full"},
            "atom_ids_b": {"widget": "text_csv_int", "group": "Group B", "tooltip_style": "dot", "layout": "full"},
            "atom_types_b": {"widget": "text_csv_str", "group": "Group B", "tooltip_style": "dot", "layout": "full"},
            "frames": {"widget": "text_csv_int", "group": "Sampling", "tooltip_style": "dot", "layout": "full"},
            "every": {"widget": "number", "group": "Sampling", "tooltip_style": "dot", "layout": "half"},
            "bins": {"widget": "number", "group": "Resolution", "tooltip_style": "dot", "layout": "half"},
            "r_max": {"widget": "number", "group": "Resolution", "tooltip_style": "dot", "layout": "half"},
            "average": {"widget": "checkbox", "group": "Output", "tooltip_style": "dot", "layout": "inline"},
            "return_stack": {"widget": "checkbox", "group": "Output", "tooltip_style": "dot", "layout": "inline"},
            "backend": {"widget": "dropdown", "group": "Engine", "tooltip_style": "dot", "layout": "half"},
        }
    },
}


def field_ui_hint(task_name: str, field_name: str) -> dict[str, Any]:
    task = TASK_UI_HINTS.get(str(task_name).lower(), {})
    fields = task.get("fields", {}) if isinstance(task, dict) else {}
    hint = fields.get(str(field_name), {}) if isinstance(fields, dict) else {}
    return dict(hint) if isinstance(hint, dict) else {}


__all__ = ["TASK_UI_HINTS", "field_ui_hint"]
