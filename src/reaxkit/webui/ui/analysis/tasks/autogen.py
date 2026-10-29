"""Auto-generated request form for analysis tasks."""

from __future__ import annotations

import json
from typing import Any, Callable

from dash import ALL, Input, Output, State, dcc, html, no_update

from reaxkit.webui.backend.api import WebUIApiService
from reaxkit.webui.ui.analysis.tasks.ui_hints import field_ui_hint
from reaxkit.webui.ui.analysis.tasks.validation import task_schema, validate_request


SelectedNodeResolver = Callable[[dict[str, Any] | None, dict[str, Any] | None], dict[str, Any] | None]

AUTO_FIELD_ID_TYPE = "analysis-auto-field"


def is_auto_task(task_name: str | None) -> bool:
    return bool(str(task_name or "").strip())


def _field_id(name: str) -> dict[str, str]:
    return {"type": AUTO_FIELD_ID_TYPE, "name": str(name)}


def _schema_field_map(schema: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    fields = schema.get("fields", []) if isinstance(schema, dict) else []
    for field in fields:
        if not isinstance(field, dict):
            continue
        name = str(field.get("name") or "").strip()
        if name:
            out[name] = field
    return out


def _initial_value(field: dict[str, Any], request: dict[str, Any]) -> Any:
    name = str(field.get("name") or "")
    default = field.get("default")
    value = request.get(name, default)
    kind = field.get("kind")
    semantic = field.get("semantic", {}) if isinstance(field.get("semantic"), dict) else {}
    choices = semantic.get("choices")

    if choices is not None and _is_list_kind(kind):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, (tuple, set)):
            return list(value)
        return [value]

    if _norm_kind(kind) == "bool":
        return ["on"] if bool(value) else []

    if _is_list_kind(kind):
        if value is None:
            return ""
        if isinstance(value, (list, tuple, set)):
            return ",".join(str(v) for v in value)
        return str(value)

    if value is None:
        return ""
    return value


def _number_step(kind: str) -> Any:
    if kind == "int":
        return 1
    if kind == "float":
        return "any"
    return None


def _norm_kind(kind: Any) -> str:
    return str(kind or "").replace(" ", "").lower()


def _is_list_kind(kind: Any) -> bool:
    norm = _norm_kind(kind)
    return "list[" in norm or "sequence[" in norm or "tuple[" in norm or "set[" in norm


def _is_list_int(kind: Any) -> bool:
    norm = _norm_kind(kind)
    return "list[int]" in norm or "sequence[int]" in norm or "tuple[int]" in norm or "set[int]" in norm


def _is_list_float(kind: Any) -> bool:
    norm = _norm_kind(kind)
    return "list[float]" in norm or "sequence[float]" in norm or "tuple[float]" in norm or "set[float]" in norm


def _is_list_str(kind: Any) -> bool:
    norm = _norm_kind(kind)
    return "list[str]" in norm or "sequence[str]" in norm or "tuple[str]" in norm or "set[str]" in norm


def _build_widget(task_name: str, field: dict[str, Any], request: dict[str, Any]) -> Any:
    name = str(field.get("name") or "")
    kind = field.get("kind")
    semantic = field.get("semantic", {}) if isinstance(field.get("semantic"), dict) else {}
    hint = field_ui_hint(task_name, name)
    widget_hint = str(hint.get("widget") or "").strip().lower()
    choices = semantic.get("choices")
    minimum = semantic.get("min")
    maximum = semantic.get("max")
    value = _initial_value(field, request)

    if widget_hint == "checkbox" or _norm_kind(kind) == "bool":
        return dcc.Checklist(
            id=_field_id(name),
            options=[{"label": "enabled", "value": "on"}],
            value=value,
            inline=True,
        )

    if widget_hint == "dropdown" and choices is not None:
        options = [{"label": str(choice), "value": choice} for choice in list(choices)]
        multi = _is_list_kind(kind)
        return dcc.Dropdown(
            id=_field_id(name),
            options=options,
            value=value,
            clearable=not multi,
            multi=multi,
        )

    if choices is not None:
        options = [{"label": str(choice), "value": choice} for choice in list(choices)]
        multi = _is_list_kind(kind)
        return dcc.Dropdown(
            id=_field_id(name),
            options=options,
            value=value,
            clearable=not multi,
            multi=multi,
        )

    if widget_hint == "slider" or (_norm_kind(kind) == "float" and minimum is not None and maximum is not None):
        try:
            min_f = float(minimum)
            max_f = float(maximum)
            val_f = float(value if value not in ("", None) else min_f)
        except Exception:
            min_f = 0.0
            max_f = 1.0
            val_f = 0.0
        step = semantic.get("step")
        try:
            step_f = float(step) if step is not None else 0.1
        except Exception:
            step_f = 0.1
        return dcc.Slider(id=_field_id(name), min=min_f, max=max_f, step=step_f, value=val_f)

    if widget_hint == "number" or _norm_kind(kind) in {"int", "float"}:
        return dcc.Input(
            id=_field_id(name),
            type="number",
            value=value,
            min=minimum,
            max=maximum,
            step=_number_step(_norm_kind(kind)),
        )

    return dcc.Input(id=_field_id(name), type="text", value=value)


def render_auto_task_form(
    lines: list[Any],
    node: dict[str, Any],
    *,
    task_name: str,
    schema: dict[str, Any] | None,
) -> Any:
    request = node.get("request", {}) if isinstance(node.get("request"), dict) else {}
    is_running = str(node.get("status", "")).lower() == "running"
    fields = schema.get("fields", []) if isinstance(schema, dict) else []

    lines.append(html.Div(f"Task: {task_name}", className="rk-subtitle"))
    groups: dict[str, list[Any]] = {}
    if not fields:
        lines.append(html.Div("No autogenerated fields available for this task schema."))
    for field in fields:
        if not isinstance(field, dict):
            continue
        name = str(field.get("name") or "").strip()
        if not name:
            continue
        hint = field_ui_hint(task_name, name)
        semantic = field.get("semantic", {}) if isinstance(field.get("semantic"), dict) else {}
        group = str(hint.get("group") or "Parameters")
        label_base = str(semantic.get("label") or name).strip() or name
        units = str(semantic.get("units") or "").strip()
        label = f"{label_base} ({units})" if units else label_base
        widget = _build_widget(task_name, field, request)
        # Memory persistence is scoped to the node; switching nodes restores drafts.
        widget.persistence = str(node.get("id"))
        widget.persistence_type = "memory"
        label_for = json.dumps(_field_id(name), sort_keys=True, separators=(",", ":"))
        groups.setdefault(group, []).append(html.Div([
            html.Label(label, htmlFor=label_for), widget,
            html.Small(str(semantic.get("help") or ""), className="rk-hint"),
        ], className="rk-field"))

    advanced = []
    for group, children in groups.items():
        section = html.Fieldset([html.Legend(group), *children], className="rk-field-group")
        if group.lower() in {"engine", "output", "advanced"}:
            advanced.append(section)
        else:
            lines.append(section)
    if advanced:
        lines.append(html.Details([html.Summary("Advanced settings"), *advanced], className="rk-advanced"))

    # Keep utility controls available for shared callbacks.
    lines.extend(
        [
            dcc.Input(id="util-filter-values", type="text", value="", style={"display": "none"}),
            dcc.Dropdown(id="util-filter-column", options=[], value=None, style={"display": "none"}),
            dcc.Dropdown(id="util-denoise-column", options=[], value=None, style={"display": "none"}),
            dcc.Input(id="util-denoise-alpha", type="number", value=0.3, style={"display": "none"}),
            dcc.Input(id="util-denoise-window", type="number", value=5, style={"display": "none"}),
            dcc.Dropdown(id="util-denoise-group", options=[], value=None, style={"display": "none"}),
            dcc.Dropdown(id="util-denoise-xcol", options=[], value=None, style={"display": "none"}),
            html.Div(id="auto-form-validation", className="rk-validation", role="status", **{"aria-live": "polite"}),
            html.Div([
                html.Button("Apply changes", id="btn-auto-apply", n_clicks=0),
                html.Button("Discard", id="btn-auto-discard", n_clicks=0),
                html.Button("Run analysis", id="btn-apply-node", n_clicks=0, disabled=is_running, className="rk-btn-exec"),
            ], className="rk-inline-actions rk-form-actions"),
        ]
    )
    return html.Div(lines, className="rk-stack")


def register_auto_task_callbacks(app, service: WebUIApiService, *, selected_node: SelectedNodeResolver) -> None:
    def draft(values, ids, session, snapshot):
        node = selected_node(snapshot, session)
        if not node or node.get("kind") != "analysis" or not ids:
            return node, {}, []
        request, errors = validate_request(task_schema(service, node), node.get("request"), ids, values)
        return node, request, errors

    @app.callback(
        Output("auto-form-validation", "children"),
        Output("btn-auto-apply", "disabled"),
        Output("btn-apply-node", "disabled"),
        Input({"type": AUTO_FIELD_ID_TYPE, "name": ALL}, "value"),
        Input("pipeline-store", "data"),
        State({"type": AUTO_FIELD_ID_TYPE, "name": ALL}, "id"),
        State("session-store", "data"),
    )
    def validate_draft(values, snapshot, ids, session):
        node, request, errors = draft(values, ids, session, snapshot)
        if not node:
            return no_update, no_update, no_update
        running = node.get("status") in {"running", "queued"}
        if errors:
            return html.Ul([html.Li(error) for error in errors]), True, True
        changed = request != node.get("request", {})
        message = "Unsaved changes. Apply to save, or run with these values." if changed else "All changes saved."
        return message, running or not changed, running

    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("btn-auto-apply", "n_clicks"),
        State({"type": AUTO_FIELD_ID_TYPE, "name": ALL}, "value"),
        State({"type": AUTO_FIELD_ID_TYPE, "name": ALL}, "id"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def apply_draft(clicks, values, ids, session, snapshot):
        if not clicks or not session:
            return no_update, no_update
        node, request, errors = draft(values, ids, session, snapshot)
        if not node or errors:
            return no_update, "Please correct the highlighted parameters."
        if node.get("status") in {"queued", "running"}:
            return no_update, "Wait for the current analysis before applying changes."
        service.update_node(session["pipeline_id"], node["id"], {"request": request})
        return service.get_pipeline(session["pipeline_id"]), "Parameters saved."

    @app.callback(
        Output({"type": AUTO_FIELD_ID_TYPE, "name": ALL}, "value"),
        Input("btn-auto-discard", "n_clicks"),
        State({"type": AUTO_FIELD_ID_TYPE, "name": ALL}, "id"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def discard_draft(clicks, ids, session, snapshot):
        node = selected_node(snapshot, session)
        if not clicks or not node:
            return [no_update for _ in ids]
        fields = _schema_field_map(task_schema(service, node))
        return [_initial_value(fields[item["name"]], node.get("request", {}))
                if item.get("name") in fields else no_update for item in ids]


__all__ = ["is_auto_task", "register_auto_task_callbacks", "render_auto_task_form"]
