"""Register properties-panel callback section for analysis UI.

This module contains a responsibility-focused subset of analysis callback
registrations extracted from `reaxkit.webui.ui.analysis.callbacks`.

**Usage context**

- Parameter title and properties panel rendering callbacks.
- Editor/control panel callback wiring for node-type specific forms.
"""

from __future__ import annotations

from typing import Any

from reaxkit.webui.backend.api import WebUIApiService
from reaxkit.webui.ui.analysis.callback_helpers import *  # noqa: F401,F403


def register_properties_callbacks(app, service: WebUIApiService) -> None:
    """Register callback bindings for this analysis UI section.

    Parameters
    -----
    app : Any
        Dash application instance used for callback decoration.
    service : WebUIApiService
        Backend service bridge for pipeline and analysis operations.

    Returns
    -----
    None
        Registers callbacks as a side effect on `app`.

    Examples
    -----
    ```python
    register_properties_callbacks(app, service)
    ```
    Sample output:
    `None`
    Meaning:
    Callback handlers for this section are attached to the Dash app.
    """
    @app.callback(
        Output("parameters-title", "children"),
        Input("session-store", "data"),
        Input("pipeline-store", "data"),
    )
    def update_parameters_title(session: dict[str, Any] | None, snapshot: dict[str, Any] | None):
        selected_id = str((session or {}).get("selected_node_id") or "")
        if selected_id == "virtual:dataset":
            return "Parameters: Dataset"
        if selected_id == "virtual:engine":
            return "Parameters: Engine"
        if selected_id == "virtual:analysis":
            return "Parameters: Analysis"
        if selected_id.startswith("virtual:utilities"):
            return "Parameters: Utilities"
        if selected_id.startswith("virtual:visualization"):
            return "Parameters: Presentation"
        node = _selected_node(snapshot, session)
        if node:
            node_id = str(node.get("id", ""))
            kind = str(node.get("kind"))
            if kind == "visualization":
                return f"Parameters: {_visualization_display_label(snapshot, node_id)}"
            if kind == "analysis":
                return f"Parameters: {_analysis_display_label(snapshot, node_id)}"
            if kind == "utility":
                return f"Parameters: {_utility_display_label(snapshot, node_id)}"
            return f"Parameters: {str(node.get('name', 'Node'))}"
        return "Parameters"

    @app.callback(
        Output("properties-content", "children"),
        Input("pipeline-store", "data"),
        Input("session-store", "data"),
        Input("result-store", "data"),
        Input("config-store", "data"),
        Input("selected-curve-store", "data"),
    )
    def render_properties(
        snapshot: dict[str, Any] | None,
        session: dict[str, Any] | None,
        result_store_in: dict[str, Any] | None,
        config_in: dict[str, Any] | None,
        selected_curve_in: dict[str, Any] | None,
    ):
        selected_id = str((session or {}).get("selected_node_id") or "")
        result_store = dict(result_store_in or {})
        config = dict(config_in or {})
        selected_curve = dict(selected_curve_in or {})

        if selected_id == "virtual:engine":
            engine_value = str(config.get("engine_name") or "autodetect")
            if engine_value not in {"autodetect", "reaxff", "ams", "lammps"}:
                engine_value = "autodetect"
            return html.Div(
                [
                    html.Label("Engine name"),
                    dcc.Dropdown(
                        id="input-engine-name",
                        options=[
                            {"label": "Autodetect", "value": "autodetect"},
                            {"label": "ReaxFF", "value": "reaxff"},
                            {"label": "AMS", "value": "ams"},
                            {"label": "LAMMPS", "value": "lammps"},
                        ],
                        value=engine_value,
                        clearable=False,
                    ),
                    html.Div(
                        [
                            html.Label("xmolout:"),
                            dcc.Input(id="input-role-xmolout", value=str(config.get("role_xmolout") or "xmolout"), type="text"),
                        ],
                        id="engine-file-roles",
                        className="rk-stack",
                    ),
                ],
                className="rk-stack",
            )

        if selected_id == "virtual:analysis":
            catalog = _catalog_payload(service)
            task_names = [str(name) for name in (catalog.get("analysis_tasks", []) if isinstance(catalog, dict) else [])]
            task_labels = {
                str(name): str(label)
                for name, label in (
                    (catalog.get("analysis_task_labels", {}) if isinstance(catalog, dict) else {}).items()
                )
            }
            analysis_options, analysis_default = _analysis_dropdown_options(task_names, task_labels)
            return html.Div(
                [
                    html.Label("Analysis type"),
                    dcc.Dropdown(
                        id="input-analysis-type",
                        options=analysis_options,
                        value=analysis_default,
                        clearable=False,
                        closeOnSelect=True,
                    ),
                    html.Button("Add Analysis Node", id="btn-add-analysis-node", n_clicks=0),
                ],
                className="rk-stack",
            )

        if selected_id.startswith("virtual:utilities"):
            catalog = _catalog_payload(service)
            utility_specs = _utility_specs_map(catalog)
            join_label = str(utility_specs.get("join_tables", {}).get("label") or "Join tables")
            ema_label = str(utility_specs.get("denoise_ema", {}).get("label") or "Denoise (EMA)")
            sma_label = str(utility_specs.get("denoise_sma", {}).get("label") or "Denoise (SMA)")
            transform_label = str(utility_specs.get("column_transform", {}).get("label") or "Column transform")
            return html.Div(
                [
                    html.Div("Node: Utilities"),
                    html.Button(join_label, id="btn-add-join-node", n_clicks=0),
                    html.Button(ema_label, id="btn-add-ema-node", n_clicks=0),
                    html.Button(sma_label, id="btn-add-sma-node", n_clicks=0),
                    html.Button(transform_label, id="btn-add-transform-node", n_clicks=0),
                ],
                className="rk-stack",
            )

        if selected_id.startswith("virtual:visualization"):
            snapshot_nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
            draft_type = str(config.get("draft_viz_type") or "plot2d")
            cols: list[str] = ["x", "y", "value"]
            numeric_cols: list[str] = []
            if isinstance(snapshot_nodes, dict):
                aid = selected_id.split(":", 2)[2] if ":" in selected_id else None
                if aid:
                    # Best effort: infer from latest artifact in the selected analysis subtree.
                    source_art = _find_source_artifact(snapshot, aid, result_store)
                    rows = _artifact_rows(source_art if isinstance(source_art, dict) else None)
                    if rows:
                        cols = infer_columns(rows)
                        numeric_cols = infer_numeric_columns(rows)
            opts = [{"label": c, "value": c} for c in cols]
            x_default = "iter" if "iter" in cols else ("frame_index" if "frame_index" in cols else (cols[0] if cols else ""))
            y_default = next((col for col in numeric_cols if col != x_default), numeric_cols[0] if numeric_cols else (cols[1] if len(cols) > 1 else x_default))
            hist_default = numeric_cols[0] if numeric_cols else (cols[0] if cols else "")
            body: list[Any] = [
                html.Label("Presentation type"),
                dcc.Dropdown(
                    id="viz-type",
                    options=[
                        {"label": "plot2d", "value": "plot2d"},
                        {"label": "histogram", "value": "histogram"},
                        {"label": "scatter3d", "value": "scatter3d"},
                        {"label": "table", "value": "table"},
                    ],
                    value=draft_type,
                    clearable=False,
                ),
            ]
            if draft_type == "plot2d":
                body.extend(
                    [
                        html.Label("x axis content"),
                        dcc.Dropdown(id="viz-x-col", options=opts, value=x_default, clearable=False),
                        html.Label("y axis content"),
                        dcc.Dropdown(id="viz-y-col", options=opts, value=y_default, clearable=False),
                        html.Label("group content"),
                        dcc.Dropdown(id="viz-group-col", options=opts, value=None, clearable=True),
                        html.Label("group aggregation"),
                        dcc.Dropdown(
                            id="viz-group-agg",
                            options=[
                                {"label": "none", "value": "none"},
                                {"label": "mean", "value": "mean"},
                                {"label": "median", "value": "median"},
                                {"label": "min", "value": "min"},
                                {"label": "max", "value": "max"},
                                {"label": "sum", "value": "sum"},
                                {"label": "count", "value": "count"},
                                {"label": "std", "value": "std"},
                            ],
                            value="none",
                            clearable=False,
                        ),
                        html.Label("Row filters"),
                        dash_table.DataTable(
                            id="viz-row-filters",
                            columns=[
                                {"name": "Column", "id": "column", "presentation": "dropdown"},
                                {"name": "Operator", "id": "op", "presentation": "dropdown"},
                                {"name": "Value", "id": "value"},
                            ],
                            data=[{"column": "", "op": "==", "value": ""}],
                            editable=True,
                            row_deletable=True,
                            dropdown=_row_filter_dropdown(cols),
                        ),
                        html.Button("Add filter", id="btn-viz-add-row-filter", n_clicks=0, className="rk-btn-save"),
                        dcc.Dropdown(id="viz-z-col", options=opts, value=x_default, clearable=False, style={"display": "none"}),
                        dcc.Dropdown(id="viz-color-col", options=opts, value=None, clearable=True, style={"display": "none"}),
                    ]
                )
            elif draft_type == "scatter3d":
                body.extend(
                    [
                        html.Label("x axis content"),
                        dcc.Dropdown(id="viz-x-col", options=opts, value=x_default, clearable=False),
                        html.Label("y axis content"),
                        dcc.Dropdown(id="viz-y-col", options=opts, value=y_default, clearable=False),
                        html.Label("z axis content"),
                        dcc.Dropdown(id="viz-z-col", options=opts, value=(cols[2] if len(cols) > 2 else x_default), clearable=False),
                        html.Label("color by"),
                        dcc.Dropdown(id="viz-color-col", options=opts, value=None, clearable=True),
                        html.Label("Row filters"),
                        dash_table.DataTable(
                            id="viz-row-filters",
                            columns=[
                                {"name": "Column", "id": "column", "presentation": "dropdown"},
                                {"name": "Operator", "id": "op", "presentation": "dropdown"},
                                {"name": "Value", "id": "value"},
                            ],
                            data=[{"column": "", "op": "==", "value": ""}],
                            editable=True,
                            row_deletable=True,
                            dropdown=_row_filter_dropdown(cols),
                        ),
                        html.Button("Add filter", id="btn-viz-add-row-filter", n_clicks=0, className="rk-btn-save"),
                        dcc.Dropdown(id="viz-group-col", options=opts, value=None, clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-group-agg", options=[{"label": "none", "value": "none"}], value="none", clearable=False, style={"display": "none"}),
                    ]
                )
            elif draft_type == "histogram":
                body.extend(
                    [
                        html.Label("value content"),
                        dcc.Dropdown(id="viz-x-col", options=opts, value=hist_default, clearable=False),
                        html.Label("Row filters"),
                        dash_table.DataTable(
                            id="viz-row-filters",
                            columns=[
                                {"name": "Column", "id": "column", "presentation": "dropdown"},
                                {"name": "Operator", "id": "op", "presentation": "dropdown"},
                                {"name": "Value", "id": "value"},
                            ],
                            data=[{"column": "", "op": "==", "value": ""}],
                            editable=True,
                            row_deletable=True,
                            dropdown=_row_filter_dropdown(cols),
                        ),
                        html.Button("Add filter", id="btn-viz-add-row-filter", n_clicks=0, className="rk-btn-save"),
                        dcc.Dropdown(id="viz-y-col", options=opts, value="", clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-z-col", options=opts, value="", clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-color-col", options=opts, value=None, clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-group-col", options=opts, value=None, clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-group-agg", options=[{"label": "none", "value": "none"}], value="none", clearable=False, style={"display": "none"}),
                    ]
                )
            else:
                body.extend(
                    [
                        dash_table.DataTable(
                            id="viz-row-filters",
                            columns=[
                                {"name": "Column", "id": "column", "presentation": "dropdown"},
                                {"name": "Operator", "id": "op", "presentation": "dropdown"},
                                {"name": "Value", "id": "value"},
                            ],
                            data=[{"column": "", "op": "==", "value": ""}],
                            editable=True,
                            row_deletable=True,
                            dropdown=_row_filter_dropdown(cols),
                            style_table={"display": "none"},
                        ),
                        html.Button("Add filter", id="btn-viz-add-row-filter", n_clicks=0, style={"display": "none"}),
                        dcc.Dropdown(id="viz-x-col", options=opts, value="", clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-y-col", options=opts, value="", clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-z-col", options=opts, value="", clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-color-col", options=opts, value=None, clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-group-col", options=opts, value=None, clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-group-agg", options=[{"label": "none", "value": "none"}], value="none", clearable=False, style={"display": "none"}),
                    ]
                )
            body.extend(
                [
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="viz-line-color-name",
                                options=[
                                    {"label": "blue", "value": "blue"},
                                    {"label": "red", "value": "red"},
                                    {"label": "black", "value": "black"},
                                    {"label": "green", "value": "green"},
                                    {"label": "orange", "value": "orange"},
                                    {"label": "purple", "value": "purple"},
                                ],
                                value="blue",
                                clearable=False,
                                style={"display": "none"},
                            )
                        ],
                        id="viz-color-name-wrap",
                        style={"display": "none"},
                    ),
                    dcc.Input(id="viz-line-color-rgb", type="text", value="", style={"display": "none"}),
                    dcc.Input(id="viz-line-width", type="number", value=2, style={"display": "none"}),
                    dcc.Dropdown(id="viz-table-filter-col", options=opts, value=None, clearable=True, style={"display": "none"}),
                    dcc.Input(id="viz-table-filter-value", type="text", value="", style={"display": "none"}),
                    dcc.Input(id="viz-table-max-rows", type="number", value=200, style={"display": "none"}),
                    dcc.Input(id="viz-font-size", type="number", value=12, style={"display": "none"}),
                    dcc.Input(id="viz-marker-size", type="number", value=6, style={"display": "none"}),
                    dcc.Dropdown(
                        id="viz-theme",
                        options=_theme_options(),
                        value="plotly_white",
                        clearable=False,
                        style={"display": "none"},
                    ),
                    dcc.Input(id="viz-axis-title-size", type="number", value=13, style={"display": "none"}),
                    dcc.Checklist(
                        id="viz-grid-on",
                        options=[{"label": "show grid", "value": "on"}],
                        value=["on"],
                        style={"display": "none"},
                    ),
                    dcc.Checklist(
                        id="viz-axis-box-on",
                        options=[{"label": "axis box on", "value": "on"}],
                        value=["on"],
                        style={"display": "none"},
                    ),
                    dcc.Input(id="viz-axis-box-width", type="number", value=1.5, style={"display": "none"}),
                    dcc.Dropdown(
                        id="viz-log-scale",
                        options=[
                            {"label": "none", "value": "none"},
                            {"label": "x", "value": "x"},
                            {"label": "y", "value": "y"},
                            {"label": "both", "value": "both"},
                        ],
                        value="none",
                        clearable=False,
                        style={"display": "none"},
                    ),
                    dcc.Input(id="viz-tick-spacing-x", type="number", value=None, style={"display": "none"}),
                    dcc.Input(id="viz-tick-spacing-y", type="number", value=None, style={"display": "none"}),
                    dcc.Checklist(id="viz-show-markers", options=[{"label": "show markers", "value": "on"}], value=[], style={"display": "none"}),
                    dcc.Checklist(id="viz-show-legend", options=[{"label": "show legend", "value": "on"}], value=["on"], style={"display": "none"}),
                    dcc.Checklist(id="viz-use-plot-title", options=[{"label": "use custom plot title", "value": "on"}], value=[], style={"display": "none"}),
                    dcc.Input(id="viz-plot-title", type="text", value="", style={"display": "none"}),
                    dcc.Input(id="viz-x-title", type="text", value="", style={"display": "none"}),
                    dcc.Input(id="viz-y-title", type="text", value="", style={"display": "none"}),
                    dcc.Input(id="viz-z-title", type="text", value="", style={"display": "none"}),
                    dcc.Dropdown(
                        id="viz-legend-position",
                        options=[
                            {"label": "top-right", "value": "top-right"},
                            {"label": "top-left", "value": "top-left"},
                            {"label": "bottom-right", "value": "bottom-right"},
                            {"label": "bottom-left", "value": "bottom-left"},
                            {"label": "right-outside", "value": "right-outside"},
                            {"label": "hidden", "value": "hidden"},
                        ],
                        value="top-right",
                        clearable=False,
                        style={"display": "none"},
                    ),
                    html.Button("Add Presentation", id="btn-add-visualization-node", n_clicks=0, className="rk-btn-exec"),
                ]
            )
            return html.Div(body, className="rk-stack")

        if selected_id == "virtual:dataset":
            return html.Div(
                [
                    html.Label("Dataset path"),
                    dcc.Input(id="input-dataset-path", value=str(config.get("dataset_path") or os.getcwd()), type="text"),
                    html.Button("Browse...", id="btn-browse-dataset", n_clicks=0),
                    html.Button("Load Dataset", id="btn-load-dataset", n_clicks=0),
                    html.Hr(),
                    html.Label("ReaxKit workspace"),
                    dcc.Checklist(
                        id="input-default-workspace",
                        options=[{"label": "Default workspace", "value": "default"}],
                        value=["default"] if bool(config.get("workspace_default", True)) else [],
                    ),
                    dcc.Input(
                        id="input-workspace-dir",
                        value=str(config.get("workspace_dir") or _default_workspace_dir_for_dataset(config.get("dataset_path"))),
                        type="text",
                    ),
                    html.Div(
                        [
                            dcc.Input(id="input-snapshot-path", value="./reaxkit.pipeline.json", type="text"),
                            html.Button("Save Snapshot", id="btn-save-snapshot", n_clicks=0),
                            html.Button("Load Snapshot", id="btn-load-snapshot", n_clicks=0),
                            dcc.Input(id="input-bundle-dir", value="./reaxkit.bundle", type="text"),
                            html.Button("Export Bundle", id="btn-export-bundle", n_clicks=0),
                        ],
                        style={"display": "none"},
                    ),
                ],
                className="rk-stack",
            )

        node = _selected_node(snapshot, session)
        if not node:
            return "Select a pipeline node."

        lines: list[Any] = [html.Div(f"Status: {node.get('status', 'idle')}")]
        if str(node.get("kind", "")) != "analysis":
            lines.insert(0, html.Div(f"Type: {node.get('kind', 'unknown')}"))
        if node.get("kind") == "dataset":
            meta = node.get("metadata", {})
            dataset = meta.get("dataset", {}) if isinstance(meta, dict) else {}
            sources = dataset.get("sources", {}) if isinstance(dataset, dict) else {}
            lines.extend(
                [
                    html.Div(f"engine: {dataset.get('engine_override') or dataset.get('engine_detected') or 'unknown'}"),
                    html.Div(f"trajectory: {sources.get('trajectory', '(unset)')}"),
                    html.Div(f"bonds: {sources.get('bonds', '(unset)')}"),
                ]
            )
            return html.Div(lines, className="rk-stack")

        if node.get("kind") == "analysis":
            req = node.get("request", {}) if isinstance(node.get("request", {}), dict) else {}
            task_name = str(node.get("metadata", {}).get("task_name") or node.get("name") or "").strip()
            catalog = _catalog_payload(service)
            schemas = _analysis_schema_map(catalog)
            schema = schemas.get(task_name, {})
            task_view = render_task_analysis_properties(node, lines, schema=schema if isinstance(schema, dict) else None)
            if task_view is not None:
                return task_view
            fields = schema.get("fields", []) if isinstance(schema, dict) else []
            is_running = str(node.get("status", "")).lower() == "running"
            schema_lines: list[Any] = []
            for field in fields:
                if not isinstance(field, dict):
                    continue
                fname = str(field.get("name") or "").strip()
                if not fname:
                    continue
                ftype = str(field.get("type") or "Any")
                fdefault = field.get("default")
                semantic = field.get("semantic", {}) if isinstance(field.get("semantic"), dict) else {}
                semantic_bits: list[str] = []
                if "help" in semantic and semantic.get("help"):
                    semantic_bits.append(f"help={semantic.get('help')}")
                if "choices" in semantic and semantic.get("choices") is not None:
                    semantic_bits.append(f"choices={semantic.get('choices')!r}")
                if "min" in semantic and semantic.get("min") is not None:
                    semantic_bits.append(f"min={semantic.get('min')!r}")
                if "max" in semantic and semantic.get("max") is not None:
                    semantic_bits.append(f"max={semantic.get('max')!r}")
                if "units" in semantic and semantic.get("units"):
                    semantic_bits.append(f"units={semantic.get('units')}")
                schema_lines.append(
                    html.Div(
                        [
                            html.Code(fname),
                            html.Span(f" : {ftype}"),
                            html.Span(f"  default={fdefault!r}" if fdefault is not None else "  default=None"),
                            html.Span(f"  [{'; '.join(semantic_bits)}]" if semantic_bits else ""),
                        ]
                    )
                )
            request_json = json.dumps(req, indent=2, ensure_ascii=True)
            lines.extend(
                [
                    html.Div(f"Task: {task_name or 'analysis'}"),
                    html.Div("Edit request JSON", className="rk-subtitle"),
                    dcc.Textarea(
                        id="analysis-request-json",
                        value=request_json,
                        style={"width": "100%", "minHeight": "200px", "fontFamily": "Consolas, Courier New, monospace"},
                    ),
                    html.Div("Schema", className="rk-subtitle"),
                    html.Div(schema_lines or [html.Div("No schema metadata found for this task.")], className="rk-stack"),
                    *hidden_task_inputs(),
                    dcc.Dropdown(id="util-filter-column", options=[], value=None, style={"display": "none"}),
                    dcc.Input(id="util-filter-values", type="text", value="", style={"display": "none"}),
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
                                html.Button("Save Params", id="btn-save-analysis-request-json", n_clicks=0, className="rk-btn-save"),
                                html.Button("Execute", id="btn-apply-node", n_clicks=0, className="rk-btn-exec"),
                            ],
                            className="rk-inline-actions",
                        )
                    ),
                ]
            )
            return html.Div(lines, className="rk-stack")

        if node.get("kind") == "utility":
            req = node.get("request", {}) if isinstance(node.get("request", {}), dict) else {}
            util_name = canonical_utility_name(str(node.get("metadata", {}).get("utility_name") or node.get("name") or "").lower())
            is_running = str(node.get("status", "")).lower() == "running"
            source_art = _find_source_artifact(snapshot, str(node.get("id")), result_store)
            source_rows = _artifact_rows(source_art if isinstance(source_art, dict) else None)
            cols = infer_columns(source_rows)
            numeric_cols = infer_numeric_columns(source_rows)
            default_req = default_utility_request(util_name, columns=cols, numeric_columns=numeric_cols)
            merged_req = dict(default_req)
            merged_req.update(req)
            catalog = _catalog_payload(service)
            utility_specs = _utility_specs_map(catalog)
            utility_spec = utility_specs.get(util_name, {})
            if util_name == "join_tables":
                left_source_id = str(node.get("parent_id") or "")
                left_source_art = _find_source_artifact(snapshot, left_source_id, result_store)
                left_rows = _artifact_rows(left_source_art if isinstance(left_source_art, dict) else None)
                source_options = _join_source_options(
                    snapshot,
                    exclude_node_ids={str(node.get("id") or ""), left_source_id},
                )
                selected_right_source = str(merged_req.get("right_source_node_id") or "").strip()
                valid_right_ids = {str(opt.get("value")) for opt in source_options}
                if selected_right_source not in valid_right_ids:
                    selected_right_source = ""
                right_source_art = _find_source_artifact(snapshot, selected_right_source, result_store)
                right_rows = _artifact_rows(right_source_art if isinstance(right_source_art, dict) else None)
                join_keys = _join_key_candidates(left_rows, right_rows)
                selected_keys = [key for key in _normalize_join_keys(merged_req.get("keys")) if key in set(join_keys)]
                if not selected_keys:
                    selected_keys = join_keys[: min(3, len(join_keys))]
                lines.extend(
                    [
                        html.Div(
                            [
                                html.Button("Delete it", id="btn-delete-node", n_clicks=0, className="rk-btn-save"),
                            ],
                            className="rk-inline-actions",
                        ),
                        html.Div(f"Utility: {util_name}"),
                        html.Div(f"Left source: {_node_display_label(snapshot, left_source_id)}"),
                        html.Label("Right source"),
                        dcc.Dropdown(
                            id="util-join-right-source",
                            options=source_options,
                            value=selected_right_source or None,
                            clearable=True,
                            placeholder="Select another analysis or utility result",
                        ),
                        html.Label("Join keys"),
                        dcc.Dropdown(
                            id="util-join-keys",
                            options=[{"label": key, "value": key} for key in join_keys],
                            value=selected_keys,
                            multi=True,
                            clearable=True,
                            placeholder="Shared columns used to merge rows",
                        ),
                        html.Label("Join type"),
                        dcc.Dropdown(
                            id="util-join-how",
                            options=[
                                {"label": "inner", "value": "inner"},
                                {"label": "left", "value": "left"},
                                {"label": "outer", "value": "outer"},
                            ],
                            value=str(merged_req.get("how") or "inner"),
                            clearable=False,
                        ),
                        html.Label("Left suffix"),
                        dcc.Input(
                            id="util-join-left-suffix",
                            type="text",
                            value=str(merged_req.get("left_suffix") or ""),
                        ),
                        html.Label("Right suffix"),
                        dcc.Input(
                            id="util-join-right-suffix",
                            type="text",
                            value=str(merged_req.get("right_suffix") or "_right"),
                        ),
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
            field_specs = utility_spec.get("fields", []) if isinstance(utility_spec, dict) else []
            field_controls: list[Any] = []
            for field in field_specs:
                if not isinstance(field, dict):
                    continue
                fname = str(field.get("name") or "").strip()
                if not fname:
                    continue
                ftype = str(field.get("type") or "Any")
                fhelp = str(field.get("help") or "").strip()
                current_value = merged_req.get(fname, field.get("default"))
                options = _utility_field_options(
                    util_name,
                    fname,
                    columns=cols,
                    numeric_columns=numeric_cols,
                )

                control: Any
                if "list[str]" in ftype and options:
                    value = current_value if isinstance(current_value, list) else _parse_csv_strs(str(current_value or "")) or []
                    control = dcc.Dropdown(
                        id={"type": "utility-field", "name": fname},
                        options=options,
                        value=value,
                        multi=True,
                        clearable=True,
                    )
                elif options:
                    value = str(current_value or "").strip()
                    control = dcc.Dropdown(
                        id={"type": "utility-field", "name": fname},
                        options=options,
                        value=(value if value else None),
                        clearable=True,
                    )
                elif "int" in ftype or "float" in ftype:
                    control = dcc.Input(
                        id={"type": "utility-field", "name": fname},
                        type="number",
                        value=current_value,
                        debounce=True,
                        step=1 if "int" in ftype and "float" not in ftype else "any",
                    )
                else:
                    text_value = current_value if isinstance(current_value, str) else ("" if current_value is None else str(current_value))
                    control = dcc.Input(
                        id={"type": "utility-field", "name": fname},
                        type="text",
                        value=text_value,
                        debounce=True,
                    )
                field_controls.append(
                    html.Div(
                        [
                            html.Label(fname),
                            control,
                            html.Div(fhelp, className="rk-help-text") if fhelp else html.Div(style={"display": "none"}),
                        ],
                        className="rk-stack",
                    )
                )
            lines.extend(
                [
                    html.Div(
                        [
                            html.Button("Delete it", id="btn-delete-node", n_clicks=0, className="rk-btn-save"),
                        ],
                        className="rk-inline-actions",
                    ),
                    html.Div(f"Utility: {util_name}"),
                ]
            )
            if field_controls:
                lines.extend(field_controls)
                lines.append(
                    html.Div("Utility parameters auto-save on field update.", className="rk-subtitle")
                )
                lines.append(
                    html.Div("Execution in progress...")
                    if is_running
                    else html.Div(
                        [
                            html.Button("Execute", id="btn-apply-node", n_clicks=0, className="rk-btn-exec"),
                        ],
                        className="rk-inline-actions",
                    )
                )
                return html.Div(lines, className="rk-stack")

            lines.extend(
                [
                    html.Div("Edit request JSON", className="rk-subtitle"),
                    dcc.Textarea(
                        id="utility-request-json",
                        value=json.dumps(merged_req, indent=2, ensure_ascii=True),
                        style={"width": "100%", "minHeight": "180px", "fontFamily": "Consolas, Courier New, monospace"},
                    ),
                    html.Div("Schema", className="rk-subtitle"),
                    html.Div([html.Div("No schema metadata found for this utility.")], className="rk-stack"),
                    *hidden_task_inputs(),
                    (
                        html.Div("Execution in progress...")
                        if is_running
                        else html.Div(
                            [
                                html.Button("Save Params", id="btn-save-utility-request-json", n_clicks=0, className="rk-btn-save"),
                                html.Button("Execute", id="btn-apply-node", n_clicks=0, className="rk-btn-exec"),
                            ],
                            className="rk-inline-actions",
                        )
                    ),
                ]
            )
            return html.Div(lines, className="rk-stack")

        if node.get("kind") == "visualization":
            req = node.get("request", {}) if isinstance(node.get("request", {}), dict) else {}
            source_art = _find_source_artifact(snapshot, str(node.get("id")), {})
            source_rows = _artifact_rows(source_art if isinstance(source_art, dict) else None)
            cols = infer_columns(source_rows) or ["x", "y", "value"]
            numeric_cols = infer_numeric_columns(source_rows)
            x_default = "iter" if "iter" in cols else ("frame_index" if "frame_index" in cols else (cols[0] if cols else ""))
            y_default = next((col for col in numeric_cols if col != x_default), numeric_cols[0] if numeric_cols else (cols[1] if len(cols) > 1 else x_default))
            z_default = cols[2] if len(cols) > 2 else y_default
            hist_default = numeric_cols[0] if numeric_cols else (cols[0] if cols else "")
            opts = [{"label": c, "value": c} for c in cols]
            viz_type = str(req.get("visualization_type") or "plot2d")
            trace_styles = _trace_styles_map(req.get("trace_styles"))
            selected_curve_key = ""
            selected_curve_label = ""
            if str(viz_type) == "plot2d":
                if (
                    isinstance(selected_curve, dict)
                    and str(selected_curve.get("node_id") or "") == str(node.get("id") or "")
                ):
                    selected_curve_key = str(selected_curve.get("trace_key") or "").strip()
                    selected_curve_label = str(selected_curve.get("trace_label") or selected_curve_key).strip()
            selected_style = trace_styles.get(selected_curve_key, {}) if selected_curve_key else {}
            color_options = [
                {"label": "blue", "value": "blue"},
                {"label": "red", "value": "red"},
                {"label": "black", "value": "black"},
                {"label": "green", "value": "green"},
                {"label": "orange", "value": "orange"},
                {"label": "purple", "value": "purple"},
            ]
            theme_options = _theme_options()
            legend_options = [
                {"label": "top-right", "value": "top-right"},
                {"label": "top-left", "value": "top-left"},
                {"label": "bottom-right", "value": "bottom-right"},
                {"label": "bottom-left", "value": "bottom-left"},
                {"label": "right-outside", "value": "right-outside"},
                {"label": "hidden", "value": "hidden"},
            ]
            body: list[Any] = [
                html.Div(
                    [
                        html.Button("Delete it", id="btn-delete-node", n_clicks=0, className="rk-btn-save"),
                    ],
                    className="rk-inline-actions",
                ),
                html.Label("Presentation type"),
                dcc.Dropdown(
                    id="viz-type",
                    options=[
                        {"label": "plot2d", "value": "plot2d"},
                        {"label": "histogram", "value": "histogram"},
                        {"label": "scatter3d", "value": "scatter3d"},
                        {"label": "table", "value": "table"},
                    ],
                    value=viz_type,
                    clearable=False,
                ),
            ]

            if viz_type == "plot2d":
                body.extend(
                    [
                        html.Label("x axis content"),
                        dcc.Dropdown(id="viz-x-col", options=opts, value=str(req.get("x_col") or x_default), clearable=False),
                        html.Label("y axis content"),
                        dcc.Dropdown(id="viz-y-col", options=opts, value=str(req.get("y_col") or y_default), clearable=False),
                        html.Label("group content"),
                        dcc.Dropdown(id="viz-group-col", options=opts, value=str(req.get("group_col") or "") or None, clearable=True),
                        html.Label("group aggregation"),
                        dcc.Dropdown(
                            id="viz-group-agg",
                            options=[
                                {"label": "none", "value": "none"},
                                {"label": "mean", "value": "mean"},
                                {"label": "median", "value": "median"},
                                {"label": "min", "value": "min"},
                                {"label": "max", "value": "max"},
                                {"label": "sum", "value": "sum"},
                                {"label": "count", "value": "count"},
                                {"label": "std", "value": "std"},
                            ],
                            value=_normalize_group_agg(req.get("group_agg")),
                            clearable=False,
                        ),
                        html.Label("Row filters"),
                        dash_table.DataTable(
                            id="viz-row-filters",
                            columns=[
                                {"name": "Column", "id": "column", "presentation": "dropdown"},
                                {"name": "Operator", "id": "op", "presentation": "dropdown"},
                                {"name": "Value", "id": "value"},
                            ],
                            data=_row_filter_table_data(req.get("row_filters")),
                            editable=True,
                            row_deletable=True,
                            dropdown=_row_filter_dropdown(cols),
                        ),
                        html.Button("Add filter", id="btn-viz-add-row-filter", n_clicks=0, className="rk-btn-save"),
                        dcc.Checklist(
                            id="viz-use-plot-title",
                            options=[{"label": "use custom plot title", "value": "on"}],
                            value=["on"] if _flag_on(req.get("use_plot_title"), False) else [],
                        ),
                        html.Label("Plot title"),
                        dcc.Input(id="viz-plot-title", type="text", value=str(req.get("plot_title") or ""), debounce=True),
                        html.Label("X axis title"),
                        dcc.Input(id="viz-x-title", type="text", value=str(req.get("x_title") or ""), debounce=True),
                        html.Label("Y axis title"),
                        dcc.Input(id="viz-y-title", type="text", value=str(req.get("y_title") or ""), debounce=True),
                        dcc.Input(id="viz-z-title", type="text", value=str(req.get("z_title") or ""), debounce=True, style={"display": "none"}),
                        html.Div("Level 1 - simple controls", className="rk-subtitle"),
                        html.Div(
                            [
                                html.Label("Color"),
                                dcc.Dropdown(
                                    id="viz-line-color-name",
                                    options=color_options,
                                    value=str(req.get("line_color") or "blue"),
                                    clearable=False,
                                ),
                            ],
                            id="viz-color-name-wrap",
                            className="rk-stack",
                        ),
                        html.Label("Color (RGB, optional)"),
                        dcc.Input(id="viz-line-color-rgb", type="text", value=str(req.get("line_color_rgb") or ""), placeholder="e.g. rgb(255,0,0)"),
                        html.Label("Line width"),
                        dcc.Input(id="viz-line-width", type="number", value=float(req.get("line_width") or 2), min=1, max=8, step=1),
                        html.Label("Font size"),
                        dcc.Input(id="viz-font-size", type="number", value=float(req.get("font_size") or 12), min=8, max=30, step=1),
                        html.Label("Marker size"),
                        dcc.Input(id="viz-marker-size", type="number", value=float(req.get("marker_size") or 6), min=0, max=20, step=1),
                        dcc.Checklist(
                            id="viz-show-markers",
                            options=[{"label": "show markers", "value": "on"}],
                            value=["on"] if _flag_on(req.get("show_markers"), False) else [],
                        ),
                        dcc.Checklist(
                            id="viz-show-legend",
                            options=[{"label": "show legend", "value": "on"}],
                            value=["on"] if _flag_on(req.get("show_legend"), True) else [],
                        ),
                        html.Label("Theme"),
                        dcc.Dropdown(id="viz-theme", options=theme_options, value=str(req.get("theme") or "plotly_white"), clearable=False),
                        html.Div("Curve settings", className="rk-subtitle"),
                        html.Div(
                            f"Selected curve: {selected_curve_label}" if selected_curve_key else "Click any point on a curve to edit curve-specific style.",
                            className="rk-help-inline",
                        ),
                        dcc.Input(id="viz-selected-trace-key", type="text", value=selected_curve_key, style={"display": "none"}),
                        html.Label("Curve color"),
                        dcc.Dropdown(
                            id="viz-curve-color-name",
                            options=color_options,
                            value=str(selected_style.get("line_color") or "blue"),
                            clearable=False,
                            disabled=not bool(selected_curve_key),
                        ),
                        html.Label("Curve color (RGB, optional)"),
                        dcc.Input(
                            id="viz-curve-color-rgb",
                            type="text",
                            value=str(selected_style.get("line_color_rgb") or ""),
                            placeholder="e.g. rgb(255,0,0)",
                            disabled=not bool(selected_curve_key),
                        ),
                        html.Label("Curve line width"),
                        dcc.Input(
                            id="viz-curve-line-width",
                            type="number",
                            value=float(selected_style.get("line_width") or req.get("line_width") or 2),
                            min=1,
                            max=8,
                            step=1,
                            disabled=not bool(selected_curve_key),
                        ),
                        html.Label("Curve marker size"),
                        dcc.Input(
                            id="viz-curve-marker-size",
                            type="number",
                            value=float(selected_style.get("marker_size") or req.get("marker_size") or 6),
                            min=0,
                            max=20,
                            step=1,
                            disabled=not bool(selected_curve_key),
                        ),
                        dcc.Checklist(
                            id="viz-curve-show-markers",
                            options=[{"label": "show markers for this curve", "value": "on"}],
                            value=["on"] if _flag_on(selected_style.get("show_markers"), False) else [],
                            style={} if selected_curve_key else {"display": "none"},
                        ),
                        html.Details(
                            [
                                html.Summary("Advanced Settings"),
                                html.Div(
                                    [
                                        html.Label("axis title size"),
                                        dcc.Input(id="viz-axis-title-size", type="number", value=float(req.get("axis_title_size") or 13), min=8, max=40, step=1),
                                        dcc.Checklist(
                                            id="viz-grid-on",
                                            options=[{"label": "grid on", "value": "on"}],
                                            value=["on"] if _flag_on(req.get("grid_on"), True) else [],
                                        ),
                                        dcc.Checklist(
                                            id="viz-axis-box-on",
                                            options=[{"label": "axis box on", "value": "on"}],
                                            value=["on"] if _flag_on(req.get("axis_box_on"), True) else [],
                                        ),
                                        html.Label("axis box width"),
                                        dcc.Input(id="viz-axis-box-width", type="number", value=float(req.get("axis_box_width") or 1.5), min=0, max=8, step=0.5),
                                        html.Label("log scale"),
                                        dcc.Dropdown(
                                            id="viz-log-scale",
                                            options=[
                                                {"label": "none", "value": "none"},
                                                {"label": "x", "value": "x"},
                                                {"label": "y", "value": "y"},
                                                {"label": "both", "value": "both"},
                                            ],
                                            value=str(req.get("log_scale") or "none"),
                                            clearable=False,
                                        ),
                                        html.Label("tick spacing (x)"),
                                        dcc.Input(id="viz-tick-spacing-x", type="number", value=_parse_float(req.get("tick_spacing_x"), None), step=1),
                                        html.Label("tick spacing (y)"),
                                        dcc.Input(id="viz-tick-spacing-y", type="number", value=_parse_float(req.get("tick_spacing_y"), None), step=1),
                                        html.Label("legend position"),
                                        dcc.Dropdown(id="viz-legend-position", options=legend_options, value=str(req.get("legend_position") or "top-right"), clearable=False),
                                    ],
                                    className="rk-stack",
                                ),
                            ]
                        ),
                        dcc.Dropdown(id="viz-z-col", options=opts, value=str(req.get("z_col") or ""), clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-color-col", options=opts, value=str(req.get("color_col") or "") or None, clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-table-filter-col", options=opts, value=None, clearable=True, style={"display": "none"}),
                        dcc.Input(id="viz-table-filter-value", type="text", value="", style={"display": "none"}),
                        dcc.Input(id="viz-table-max-rows", type="number", value=200, style={"display": "none"}),
                    ]
                )
            elif viz_type == "scatter3d":
                body.extend(
                    [
                        html.Label("x axis content"),
                        dcc.Dropdown(id="viz-x-col", options=opts, value=str(req.get("x_col") or x_default), clearable=False),
                        html.Label("y axis content"),
                        dcc.Dropdown(id="viz-y-col", options=opts, value=str(req.get("y_col") or y_default), clearable=False),
                        html.Label("z axis content"),
                        dcc.Dropdown(id="viz-z-col", options=opts, value=str(req.get("z_col") or z_default), clearable=False),
                        html.Label("color by"),
                        dcc.Dropdown(id="viz-color-col", options=opts, value=str(req.get("color_col") or "") or None, clearable=True),
                        html.Label("Row filters"),
                        dash_table.DataTable(
                            id="viz-row-filters",
                            columns=[
                                {"name": "Column", "id": "column", "presentation": "dropdown"},
                                {"name": "Operator", "id": "op", "presentation": "dropdown"},
                                {"name": "Value", "id": "value"},
                            ],
                            data=_row_filter_table_data(req.get("row_filters")),
                            editable=True,
                            row_deletable=True,
                            dropdown=_row_filter_dropdown(cols),
                        ),
                        html.Button("Add filter", id="btn-viz-add-row-filter", n_clicks=0, className="rk-btn-save"),
                        dcc.Dropdown(id="viz-group-col", options=opts, value=str(req.get("group_col") or "") or None, clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-group-agg", options=[{"label": "none", "value": "none"}], value="none", clearable=False, style={"display": "none"}),
                        dcc.Input(id="viz-selected-trace-key", type="text", value="", style={"display": "none"}),
                        dcc.Dropdown(id="viz-curve-color-name", options=color_options, value="blue", clearable=False, style={"display": "none"}),
                        dcc.Input(id="viz-curve-color-rgb", type="text", value="", style={"display": "none"}),
                        dcc.Input(id="viz-curve-line-width", type="number", value=2, style={"display": "none"}),
                        dcc.Input(id="viz-curve-marker-size", type="number", value=6, style={"display": "none"}),
                        dcc.Checklist(id="viz-curve-show-markers", options=[{"label": "on", "value": "on"}], value=[], style={"display": "none"}),
                        dcc.Checklist(
                            id="viz-use-plot-title",
                            options=[{"label": "use custom plot title", "value": "on"}],
                            value=["on"] if _flag_on(req.get("use_plot_title"), False) else [],
                        ),
                        html.Label("Plot title"),
                        dcc.Input(id="viz-plot-title", type="text", value=str(req.get("plot_title") or ""), debounce=True),
                        html.Label("X axis title"),
                        dcc.Input(id="viz-x-title", type="text", value=str(req.get("x_title") or ""), debounce=True),
                        html.Label("Y axis title"),
                        dcc.Input(id="viz-y-title", type="text", value=str(req.get("y_title") or ""), debounce=True),
                        html.Label("Z axis title"),
                        dcc.Input(id="viz-z-title", type="text", value=str(req.get("z_title") or ""), debounce=True),
                        html.Div("Level 1 - simple controls", className="rk-subtitle"),
                        html.Div(
                            [
                                html.Label("Color (used when color by is empty)"),
                                dcc.Dropdown(id="viz-line-color-name", options=color_options, value=str(req.get("line_color") or "blue"), clearable=False),
                            ],
                            id="viz-color-name-wrap",
                            className="rk-stack",
                            style={"display": "grid"},
                        ),
                        html.Label("Color (RGB, optional)"),
                        dcc.Input(id="viz-line-color-rgb", type="text", value=str(req.get("line_color_rgb") or ""), placeholder="e.g. rgb(255,0,0)"),
                        html.Label("Font size"),
                        dcc.Input(id="viz-font-size", type="number", value=float(req.get("font_size") or 12), min=8, max=30, step=1),
                        html.Label("Marker size"),
                        dcc.Input(id="viz-marker-size", type="number", value=float(req.get("marker_size") or 6), min=1, max=30, step=1),
                        dcc.Checklist(
                            id="viz-show-markers",
                            options=[{"label": "show markers", "value": "on"}],
                            value=["on"] if _flag_on(req.get("show_markers"), True) else [],
                        ),
                        dcc.Checklist(
                            id="viz-show-legend",
                            options=[{"label": "show legend", "value": "on"}],
                            value=["on"] if _flag_on(req.get("show_legend"), True) else [],
                        ),
                        html.Label("Theme"),
                        dcc.Dropdown(id="viz-theme", options=theme_options, value=str(req.get("theme") or "plotly_white"), clearable=False),
                        html.Details(
                            [
                                html.Summary("Advanced Settings"),
                                html.Div(
                                    [
                                        html.Label("axis title size"),
                                        dcc.Input(id="viz-axis-title-size", type="number", value=float(req.get("axis_title_size") or 13), min=8, max=40, step=1),
                                        dcc.Checklist(
                                            id="viz-grid-on",
                                            options=[{"label": "grid on", "value": "on"}],
                                            value=["on"] if _flag_on(req.get("grid_on"), True) else [],
                                        ),
                                    ],
                                    className="rk-stack",
                                ),
                            ]
                        ),
                        dcc.Input(id="viz-line-width", type="number", value=2, style={"display": "none"}),
                        dcc.Checklist(id="viz-axis-box-on", options=[{"label": "axis box on", "value": "on"}], value=["on"], style={"display": "none"}),
                        dcc.Input(id="viz-axis-box-width", type="number", value=1.5, style={"display": "none"}),
                        dcc.Dropdown(id="viz-log-scale", options=[{"label": "none", "value": "none"}], value="none", clearable=False, style={"display": "none"}),
                        dcc.Input(id="viz-tick-spacing-x", type="number", value=None, style={"display": "none"}),
                        dcc.Input(id="viz-tick-spacing-y", type="number", value=None, style={"display": "none"}),
                        dcc.Dropdown(id="viz-legend-position", options=legend_options, value=str(req.get("legend_position") or "top-right"), clearable=False, style={"display": "none"}),
                        dcc.Dropdown(id="viz-table-filter-col", options=opts, value=None, clearable=True, style={"display": "none"}),
                        dcc.Input(id="viz-table-filter-value", type="text", value="", style={"display": "none"}),
                        dcc.Input(id="viz-table-max-rows", type="number", value=200, style={"display": "none"}),
                    ]
                )
            elif viz_type == "histogram":
                body.extend(
                    [
                        html.Label("value content"),
                        dcc.Dropdown(id="viz-x-col", options=opts, value=str(req.get("x_col") or hist_default), clearable=False),
                        html.Label("Row filters"),
                        dash_table.DataTable(
                            id="viz-row-filters",
                            columns=[
                                {"name": "Column", "id": "column", "presentation": "dropdown"},
                                {"name": "Operator", "id": "op", "presentation": "dropdown"},
                                {"name": "Value", "id": "value"},
                            ],
                            data=_row_filter_table_data(req.get("row_filters")),
                            editable=True,
                            row_deletable=True,
                            dropdown=_row_filter_dropdown(cols),
                        ),
                        html.Button("Add filter", id="btn-viz-add-row-filter", n_clicks=0, className="rk-btn-save"),
                        dcc.Dropdown(id="viz-y-col", options=opts, value=str(req.get("y_col") or ""), clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-z-col", options=opts, value=str(req.get("z_col") or ""), clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-color-col", options=opts, value=str(req.get("color_col") or "") or None, clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-group-col", options=opts, value=str(req.get("group_col") or "") or None, clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-group-agg", options=[{"label": "none", "value": "none"}], value="none", clearable=False, style={"display": "none"}),
                        dcc.Input(id="viz-selected-trace-key", type="text", value="", style={"display": "none"}),
                        dcc.Dropdown(id="viz-curve-color-name", options=color_options, value="blue", clearable=False, style={"display": "none"}),
                        dcc.Input(id="viz-curve-color-rgb", type="text", value="", style={"display": "none"}),
                        dcc.Input(id="viz-curve-line-width", type="number", value=2, style={"display": "none"}),
                        dcc.Input(id="viz-curve-marker-size", type="number", value=6, style={"display": "none"}),
                        dcc.Checklist(id="viz-curve-show-markers", options=[{"label": "on", "value": "on"}], value=[], style={"display": "none"}),
                        dcc.Checklist(
                            id="viz-use-plot-title",
                            options=[{"label": "use custom plot title", "value": "on"}],
                            value=["on"] if _flag_on(req.get("use_plot_title"), False) else [],
                        ),
                        html.Label("Plot title"),
                        dcc.Input(id="viz-plot-title", type="text", value=str(req.get("plot_title") or ""), debounce=True),
                        html.Label("X axis title"),
                        dcc.Input(id="viz-x-title", type="text", value=str(req.get("x_title") or ""), debounce=True),
                        html.Label("Y axis title"),
                        dcc.Input(id="viz-y-title", type="text", value=str(req.get("y_title") or ""), debounce=True),
                        dcc.Input(id="viz-z-title", type="text", value=str(req.get("z_title") or ""), debounce=True, style={"display": "none"}),
                        html.Div("Level 1 - simple controls", className="rk-subtitle"),
                        html.Div(
                            [
                                html.Label("Color"),
                                dcc.Dropdown(id="viz-line-color-name", options=color_options, value=str(req.get("line_color") or "blue"), clearable=False),
                            ],
                            id="viz-color-name-wrap",
                            className="rk-stack",
                            style={"display": "grid"},
                        ),
                        html.Label("Color (RGB, optional)"),
                        dcc.Input(id="viz-line-color-rgb", type="text", value=str(req.get("line_color_rgb") or ""), placeholder="e.g. rgb(255,0,0)"),
                        html.Label("Font size"),
                        dcc.Input(id="viz-font-size", type="number", value=float(req.get("font_size") or 12), min=8, max=30, step=1),
                        html.Label("Theme"),
                        dcc.Dropdown(id="viz-theme", options=theme_options, value=str(req.get("theme") or "plotly_white"), clearable=False),
                        dcc.Checklist(
                            id="viz-show-markers",
                            options=[{"label": "show markers", "value": "on"}],
                            value=[],
                            style={"display": "none"},
                        ),
                        dcc.Checklist(
                            id="viz-show-legend",
                            options=[{"label": "show legend", "value": "on"}],
                            value=["on"] if _flag_on(req.get("show_legend"), True) else [],
                        ),
                        html.Details(
                            [
                                html.Summary("Advanced Settings"),
                                html.Div(
                                    [
                                        html.Label("axis title size"),
                                        dcc.Input(id="viz-axis-title-size", type="number", value=float(req.get("axis_title_size") or 13), min=8, max=40, step=1),
                                        dcc.Checklist(
                                            id="viz-grid-on",
                                            options=[{"label": "grid on", "value": "on"}],
                                            value=["on"] if _flag_on(req.get("grid_on"), True) else [],
                                        ),
                                        dcc.Checklist(
                                            id="viz-axis-box-on",
                                            options=[{"label": "axis box on", "value": "on"}],
                                            value=["on"] if _flag_on(req.get("axis_box_on"), True) else [],
                                        ),
                                        html.Label("axis box width"),
                                        dcc.Input(id="viz-axis-box-width", type="number", value=float(req.get("axis_box_width") or 1.5), min=0, max=8, step=0.5),
                                        html.Label("log scale"),
                                        dcc.Dropdown(
                                            id="viz-log-scale",
                                            options=[
                                                {"label": "none", "value": "none"},
                                                {"label": "y", "value": "y"},
                                            ],
                                            value=str(req.get("log_scale") or "none"),
                                            clearable=False,
                                        ),
                                        html.Label("tick spacing (x)"),
                                        dcc.Input(id="viz-tick-spacing-x", type="number", value=_parse_float(req.get("tick_spacing_x"), None), step=1),
                                        html.Label("tick spacing (y)"),
                                        dcc.Input(id="viz-tick-spacing-y", type="number", value=_parse_float(req.get("tick_spacing_y"), None), step=1),
                                    ],
                                    className="rk-stack",
                                ),
                            ]
                        ),
                        dcc.Input(id="viz-marker-size", type="number", value=6, style={"display": "none"}),
                        dcc.Input(id="viz-line-width", type="number", value=2, style={"display": "none"}),
                        dcc.Dropdown(id="viz-legend-position", options=legend_options, value="hidden", clearable=False, style={"display": "none"}),
                        dcc.Dropdown(id="viz-table-filter-col", options=opts, value=None, clearable=True, style={"display": "none"}),
                        dcc.Input(id="viz-table-filter-value", type="text", value="", style={"display": "none"}),
                        dcc.Input(id="viz-table-max-rows", type="number", value=200, style={"display": "none"}),
                    ]
                )
            else:
                body.extend(
                    [
                        html.Div("Table filtering/sorting is available directly in the Visualization Canvas.", className="rk-subtitle"),
                        dash_table.DataTable(
                            id="viz-row-filters",
                            columns=[
                                {"name": "Column", "id": "column", "presentation": "dropdown"},
                                {"name": "Operator", "id": "op", "presentation": "dropdown"},
                                {"name": "Value", "id": "value"},
                            ],
                            data=_row_filter_table_data(req.get("row_filters")),
                            editable=True,
                            row_deletable=True,
                            dropdown=_row_filter_dropdown(cols),
                            style_table={"display": "none"},
                        ),
                        html.Button("Add filter", id="btn-viz-add-row-filter", n_clicks=0, style={"display": "none"}),
                        html.Label("Visible rows"),
                        dcc.Input(
                            id="viz-table-max-rows",
                            type="number",
                            value=int(req.get("table_max_rows") or 200),
                            min=10,
                            step=10,
                        ),
                        dcc.Input(id="viz-plot-title", type="text", value=str(req.get("plot_title") or ""), style={"display": "none"}),
                        dcc.Input(id="viz-x-title", type="text", value=str(req.get("x_title") or ""), style={"display": "none"}),
                        dcc.Input(id="viz-y-title", type="text", value=str(req.get("y_title") or ""), style={"display": "none"}),
                        dcc.Input(id="viz-z-title", type="text", value=str(req.get("z_title") or ""), style={"display": "none"}),
                        dcc.Checklist(id="viz-use-plot-title", options=[{"label": "use custom plot title", "value": "on"}], value=[], style={"display": "none"}),
                        dcc.Dropdown(id="viz-table-filter-col", options=opts, value=None, clearable=True, style={"display": "none"}),
                        dcc.Input(id="viz-table-filter-value", type="text", value="", style={"display": "none"}),
                        dcc.Dropdown(id="viz-x-col", options=opts, value=str(req.get("x_col") or ""), clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-y-col", options=opts, value=str(req.get("y_col") or ""), clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-z-col", options=opts, value=str(req.get("z_col") or ""), clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-color-col", options=opts, value=str(req.get("color_col") or "") or None, clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-group-col", options=opts, value=str(req.get("group_col") or "") or None, clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-group-agg", options=[{"label": "none", "value": "none"}], value="none", clearable=False, style={"display": "none"}),
                        dcc.Input(id="viz-selected-trace-key", type="text", value="", style={"display": "none"}),
                        dcc.Dropdown(id="viz-curve-color-name", options=color_options, value="blue", clearable=False, style={"display": "none"}),
                        dcc.Input(id="viz-curve-color-rgb", type="text", value="", style={"display": "none"}),
                        dcc.Input(id="viz-curve-line-width", type="number", value=2, style={"display": "none"}),
                        dcc.Input(id="viz-curve-marker-size", type="number", value=6, style={"display": "none"}),
                        dcc.Checklist(id="viz-curve-show-markers", options=[{"label": "on", "value": "on"}], value=[], style={"display": "none"}),
                        html.Div(
                            [
                                html.Label("Line color"),
                                dcc.Dropdown(id="viz-line-color-name", options=color_options, value="blue", clearable=False),
                            ],
                            id="viz-color-name-wrap",
                            className="rk-stack",
                            style={"display": "none"},
                        ),
                        dcc.Input(id="viz-line-color-rgb", type="text", value="", style={"display": "none"}),
                        dcc.Input(id="viz-line-width", type="number", value=2, style={"display": "none"}),
                        dcc.Input(id="viz-font-size", type="number", value=12, style={"display": "none"}),
                        dcc.Input(id="viz-marker-size", type="number", value=6, style={"display": "none"}),
                        dcc.Checklist(id="viz-show-markers", options=[{"label": "show markers", "value": "on"}], value=[], style={"display": "none"}),
                        dcc.Checklist(id="viz-show-legend", options=[{"label": "show legend", "value": "on"}], value=["on"], style={"display": "none"}),
                        dcc.Dropdown(id="viz-theme", options=theme_options, value="plotly_white", clearable=False, style={"display": "none"}),
                        dcc.Input(id="viz-axis-title-size", type="number", value=13, style={"display": "none"}),
                        dcc.Checklist(id="viz-grid-on", options=[{"label": "grid on", "value": "on"}], value=["on"], style={"display": "none"}),
                        dcc.Checklist(id="viz-axis-box-on", options=[{"label": "axis box on", "value": "on"}], value=["on"], style={"display": "none"}),
                        dcc.Input(id="viz-axis-box-width", type="number", value=1.5, style={"display": "none"}),
                        dcc.Dropdown(id="viz-log-scale", options=[{"label": "none", "value": "none"}], value="none", clearable=False, style={"display": "none"}),
                        dcc.Input(id="viz-tick-spacing-x", type="number", value=None, style={"display": "none"}),
                        dcc.Input(id="viz-tick-spacing-y", type="number", value=None, style={"display": "none"}),
                        dcc.Dropdown(id="viz-legend-position", options=legend_options, value="top-right", clearable=False, style={"display": "none"}),
                    ]
                )
            lines.extend(
                body
            )
            lines.append(
                html.Div(
                    [
                        html.Button("Save Params", id="btn-save-viz-params", n_clicks=0, className="rk-btn-save"),
                        html.Button("Execute", id="btn-apply-node", n_clicks=0, className="rk-btn-exec"),
                    ],
                    className="rk-inline-actions",
                    style={"display": "none"},
                )
            )
            return html.Div(lines, className="rk-stack")

        lines.append("Custom editor not yet available for this node type.")
        lines.extend(
            [
                html.Hr(),
                html.Label("Snapshot path"),
                dcc.Input(id="input-snapshot-path", value="./reaxkit.pipeline.json", type="text"),
                html.Button("Save Snapshot", id="btn-save-snapshot", n_clicks=0),
                html.Button("Load Snapshot", id="btn-load-snapshot", n_clicks=0),
                html.Label("Bundle output dir"),
                dcc.Input(id="input-bundle-dir", value="./reaxkit.bundle", type="text"),
                html.Button("Export Bundle", id="btn-export-bundle", n_clicks=0),
            ]
        )
        return html.Div(lines, className="rk-stack")

