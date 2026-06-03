"""Auto-generated request form for analysis tasks."""

from __future__ import annotations

from typing import Any, Callable

from dash import ALL, Input, Output, State, dcc, html, no_update

from reaxkit.webui.backend.api import WebUIApiService
from reaxkit.core.utils.frame_utils import parse_frame_indices
from reaxkit.webui.ui.analysis.tasks.form_utils import label_with_help
from reaxkit.webui.ui.analysis.tasks.ui_hints import field_ui_hint


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


def _parse_csv_ints(raw: Any) -> list[int] | None:
    if raw is None or str(raw).strip() == "":
        return None
    out: list[int] = []
    for token in str(raw).split(","):
        tok = token.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except Exception:
            continue
    return out or None


def _parse_csv_floats(raw: Any) -> list[float] | None:
    if raw is None or str(raw).strip() == "":
        return None
    out: list[float] = []
    for token in str(raw).split(","):
        tok = token.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except Exception:
            continue
    return out or None


def _parse_csv_strs(raw: Any) -> list[str] | None:
    if raw is None or str(raw).strip() == "":
        return None
    out = [tok.strip() for tok in str(raw).split(",") if tok.strip()]
    return out or None


def _coerce_value(raw_value: Any, field: dict[str, Any]) -> Any:
    name = str(field.get("name") or "").strip()
    kind = field.get("kind")
    semantic = field.get("semantic", {}) if isinstance(field.get("semantic"), dict) else {}
    default = field.get("default")
    choices = semantic.get("choices")

    if _norm_kind(kind) == "bool":
        return bool(isinstance(raw_value, list) and ("on" in raw_value))

    if choices is not None and _is_list_kind(kind):
        if raw_value is None:
            return []
        if isinstance(raw_value, list):
            return raw_value
        return [raw_value]
    if choices is not None:
        return raw_value if raw_value not in ("", None) else default

    if _norm_kind(kind) == "int":
        if raw_value in ("", None):
            return default
        try:
            return int(raw_value)
        except Exception:
            return default

    if _norm_kind(kind) == "float":
        if raw_value in ("", None):
            return default
        try:
            return float(raw_value)
        except Exception:
            return default

    if _is_list_int(kind):
        try:
            # Accept flexible selectors for any list[int] field, e.g.
            # "1,2,3", "1 2 3", "1-40", "1:40", "1:40:1".
            val = parse_frame_indices(raw_value)
        except Exception:
            val = None
        return val if val is not None else default

    if _is_list_float(kind):
        val = _parse_csv_floats(raw_value)
        return val if val is not None else default

    if _is_list_str(kind):
        val = _parse_csv_strs(raw_value)
        return val if val is not None else default

    if raw_value in ("", None):
        return default
    return raw_value


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
    previous_group: str | None = None
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
        if group != previous_group:
            lines.append(html.Div(group, className="rk-subtitle"))
            previous_group = group
        label_base = str(semantic.get("label") or name).strip() or name
        units = str(semantic.get("units") or "").strip()
        label = f"{label_base} ({units})" if units else label_base
        lines.append(
            label_with_help(
                label,
                semantic.get("help"),
                tooltip_style=str(hint.get("tooltip_style") or "dot"),
            )
        )
        lines.append(_build_widget(task_name, field, request))

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
            (
                html.Div("Execution in progress...")
                if is_running
                else html.Div(
                    [
                        html.Button("Execute", id="btn-apply-node", n_clicks=0, className="rk-btn-exec"),
                    ],
                    className="rk-inline-actions",
                )
            ),
        ]
    )
    return html.Div(lines, className="rk-stack")


def register_auto_task_callbacks(
    app,
    service: WebUIApiService,
    *,
    selected_node: SelectedNodeResolver,
) -> None:
    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Input({"type": AUTO_FIELD_ID_TYPE, "name": ALL}, "value"),
        State({"type": AUTO_FIELD_ID_TYPE, "name": ALL}, "id"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def on_live_auto_task_params(
        values: list[Any],
        ids: list[dict[str, Any]],
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
    ):
        if not session or not snapshot or not ids:
            return no_update

        pipeline_id = str(session.get("pipeline_id", ""))
        node = selected_node(snapshot, session)
        if not pipeline_id or not isinstance(node, dict) or str(node.get("kind")) != "analysis":
            return no_update

        task_name = str(node.get("metadata", {}).get("task_name") or node.get("name") or "").lower()
        if not is_auto_task(task_name):
            return no_update

        catalog = service.get_catalog()
        schemas = catalog.get("analysis_schemas", {}) if isinstance(catalog, dict) else {}
        schema = schemas.get(task_name, {}) if isinstance(schemas, dict) else {}
        field_map = _schema_field_map(schema if isinstance(schema, dict) else {})
        if not field_map:
            return no_update

        old_req = node.get("request", {}) if isinstance(node.get("request"), dict) else {}
        request_payload = dict(old_req)
        for raw_id, raw_value in zip(ids, values):
            if not isinstance(raw_id, dict):
                continue
            name = str(raw_id.get("name") or "").strip()
            if not name or name not in field_map:
                continue
            request_payload[name] = _coerce_value(raw_value, field_map[name])

        if request_payload == old_req:
            return no_update

        service.update_node(pipeline_id, str(node["id"]), {"request": request_payload})
        return service.get_pipeline(pipeline_id)


__all__ = [
    "is_auto_task",
    "register_auto_task_callbacks",
    "render_auto_task_form",
]
