"""Register analysis callback bindings for the Web UI.

This module hosts callback registration only and delegates helper logic to
`callback_helpers`. Keeping decorators here preserves one entrypoint while
allowing helper code to evolve in smaller files.

**Usage context**

- Dash callback registration during app startup.
- Wiring analysis/service actions to UI components.
- Preserving compatibility for imports of `register_analysis_callbacks`.
"""

from __future__ import annotations

import reaxkit.webui.ui.analysis.callback_helpers as _helper
from reaxkit.webui.ui.analysis.callback_helpers import *  # noqa: F401,F403


def register_analysis_callbacks(app, service: WebUIApiService) -> None:
    """Register analysis-focused Dash callbacks.

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
    register_analysis_callbacks(app, service)
    ```
    Sample output:
    `None`
    Meaning:
    Callback handlers are attached to the Dash app instance.
    """
    _helper._SERVICE_HANDLE = service

    register_task_callbacks(
        app,
        service,
        selected_node=_selected_node,
        parse_csv_ints=_parse_csv_ints,
        parse_csv_strs=_parse_csv_strs,
    )

    @app.callback(
        Output("status-banner", "children", allow_duplicate=True),
        Input("btn-save-snapshot", "n_clicks"),
        State("session-store", "data"),
        State("input-snapshot-path", "value"),
        prevent_initial_call=True,
    )
    def on_save_snapshot(n_clicks: int, session: dict[str, Any] | None, snapshot_path: str | None):
        if not n_clicks or not session:
            return no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        if not pipeline_id:
            return "WARN: No active pipeline"
        path = str(snapshot_path or "./reaxkit.pipeline.json")
        try:
            saved = service.export_pipeline(pipeline_id, {"path": path})
        except Exception as exc:
            return f"ERROR: Save failed: {exc}"
        return f"Snapshot saved: {saved.get('path')}"

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Output("pipeline-store", "data", allow_duplicate=True),
        Output("result-store", "data", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("btn-load-snapshot", "n_clicks"),
        State("input-snapshot-path", "value"),
        prevent_initial_call=True,
    )
    def on_load_snapshot(n_clicks: int, snapshot_path: str | None):
        if not n_clicks:
            return no_update, no_update, no_update, no_update
        path = str(snapshot_path or "").strip()
        if not path:
            return no_update, no_update, no_update, "WARN: Snapshot path required"
        try:
            snapshot = service.load_pipeline_snapshot({"path": path})
        except Exception as exc:
            return no_update, no_update, no_update, f"ERROR: Load failed: {exc}"
        _ARTIFACT_OBJ_CACHE.clear()
        _ARTIFACT_ROWS_CACHE.clear()
        _PLOT_ROWS_CACHE.clear()
        _NODE_PIPELINE_CACHE.clear()
        selected = "virtual:dataset"
        session = {"pipeline_id": snapshot.get("id"), "selected_node_id": selected}
        result_cache = _result_cache_from_snapshot(snapshot)
        return session, snapshot, result_cache, f"Snapshot loaded: {path}"

    @app.callback(
        Output("config-store", "data", allow_duplicate=True),
        Input("input-dataset-path", "value"),
        State("config-store", "data"),
        prevent_initial_call=True,
    )
    def sync_dataset_path(value: str | None, config: dict[str, Any] | None):
        cfg = dict(config or {})
        dataset_path = str(value or ".")
        cfg["dataset_path"] = dataset_path
        if bool(cfg.get("workspace_default", True)):
            cfg["workspace_dir"] = _default_workspace_dir_for_dataset(dataset_path)
        return cfg

    @app.callback(
        Output("config-store", "data", allow_duplicate=True),
        Input("input-engine-name", "value"),
        Input("input-role-xmolout", "value"),
        State("config-store", "data"),
        prevent_initial_call=True,
    )
    def sync_engine_config(
        engine_name: str | None,
        role_xmolout: str | None,
        config: dict[str, Any] | None,
    ):
        cfg = dict(config or {})
        cfg["engine_name"] = str(engine_name or "autodetect")
        cfg["role_xmolout"] = str(role_xmolout or "xmolout")
        return cfg

    @app.callback(
        Output("config-store", "data", allow_duplicate=True),
        Input("input-default-workspace", "value"),
        Input("input-workspace-dir", "value"),
        State("input-dataset-path", "value"),
        State("config-store", "data"),
        prevent_initial_call=True,
    )
    def sync_workspace_config(
        default_flags: list[str] | None,
        workspace_dir: str | None,
        dataset_path: str | None,
        config: dict[str, Any] | None,
    ):
        cfg = dict(config or {})
        use_default = "default" in (default_flags or [])
        cfg["workspace_default"] = bool(use_default)
        cfg["workspace_dir"] = (
            _default_workspace_dir_for_dataset(dataset_path or cfg.get("dataset_path"))
            if use_default
            else str(workspace_dir or _default_workspace_dir_for_dataset(dataset_path or cfg.get("dataset_path")))
        )
        return cfg

    @app.callback(
        Output("status-banner", "children", allow_duplicate=True),
        Input("btn-export-bundle", "n_clicks"),
        State("session-store", "data"),
        State("input-bundle-dir", "value"),
        prevent_initial_call=True,
    )
    def on_export_bundle(n_clicks: int, session: dict[str, Any] | None, bundle_dir: str | None):
        if not n_clicks or not session:
            return no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        if not pipeline_id:
            return "WARN: No active pipeline"
        out_dir = str(bundle_dir or "./reaxkit.bundle")
        try:
            result = service.export_pipeline_bundle(
                pipeline_id,
                {"path": out_dir, "selected_node_id": session.get("selected_node_id")},
            )
        except Exception as exc:
            return f"ERROR: Export bundle failed: {exc}"
        return f"Bundle exported: {result.get('bundle_dir')}"

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Output("pipeline-store", "data", allow_duplicate=True),
        Output("config-store", "data", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("btn-load-dataset", "n_clicks"),
        State("session-store", "data"),
        State("config-store", "data"),
        prevent_initial_call=True,
    )
    def on_load_dataset(
        n_clicks: int,
        session: dict[str, Any] | None,
        config: dict[str, Any] | None,
    ):
        if not n_clicks or not session or "pipeline_id" not in session:
            return no_update, no_update, no_update, no_update

        pipeline_id = str(session["pipeline_id"])
        cfg = dict(config or {})
        dataset_path = str(cfg.get("dataset_path") or ".")
        engine_name = str(cfg.get("engine_name") or "autodetect")
        role_xmolout = str(cfg.get("role_xmolout") or "xmolout")
        workspace_default = bool(cfg.get("workspace_default", True))
        run_dir = str(dataset_path or ".").strip() or "."
        workspace_dir = (
            _default_workspace_dir_for_dataset(run_dir)
            if workspace_default
            else str(cfg.get("workspace_dir") or _default_workspace_dir_for_dataset(run_dir))
        )
        engine_value = str(engine_name or "autodetect").strip().lower()
        forced_engine = None if engine_value == "autodetect" else engine_value
        sources = {"trajectory": "xmolout"}
        if engine_value == "reaxff":
            sources["trajectory"] = str(role_xmolout or "xmolout").strip() or "xmolout"

        dataset_node = service.load_dataset(
            pipeline_id,
            {
                "run_dir": run_dir,
                "engine": forced_engine,
                "sources": sources,
                "project_root": workspace_dir,
            },
        )
        snapshot = service.get_pipeline(pipeline_id)
        loaded_dataset = _latest_dataset_node(snapshot) or dataset_node
        dataset_meta = loaded_dataset.get("metadata", {}) if isinstance(loaded_dataset, dict) else {}
        dataset_payload = dataset_meta.get("dataset", {}) if isinstance(dataset_meta, dict) else {}
        detected_engine = str(dataset_payload.get("engine_override") or dataset_payload.get("engine_detected") or "unknown")
        next_cfg = dict(cfg)
        next_cfg["dataset_path"] = run_dir
        next_cfg["workspace_dir"] = workspace_dir
        next_cfg["role_xmolout"] = str(role_xmolout or "xmolout")
        if engine_value != "autodetect":
            next_cfg["engine_name"] = engine_value
        elif detected_engine != "unknown":
            next_cfg["engine_name"] = "autodetect"
        return {"pipeline_id": pipeline_id, "selected_node_id": "virtual:dataset"}, snapshot, next_cfg, "Dataset loaded"

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Output("pipeline-store", "data", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("btn-add-analysis-node", "n_clicks"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        State("input-analysis-type", "value"),
        prevent_initial_call=True,
    )
    def on_add_analysis_node(
        n_clicks: int,
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
        analysis_type: str | None,
    ):
        if not n_clicks or not session or not snapshot:
            return no_update, no_update, no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        if not pipeline_id:
            return no_update, no_update, "WARN: No active pipeline"

        dataset_node = _latest_dataset_node(snapshot)
        if not dataset_node:
            return no_update, no_update, "WARN: Load a dataset first"

        catalog = _catalog_payload(service)
        task_names = [str(name) for name in (catalog.get("analysis_tasks", []) if isinstance(catalog, dict) else [])]
        task = str(analysis_type or "").strip().lower()
        if not task:
            task = task_names[0] if task_names else "msd"
        if task_names and task not in set(task_names):
            return no_update, no_update, f"WARN: Unknown analysis task '{task}'"

        request_defaults = _analysis_defaults_for_task(task, catalog)
        analysis_node = service.add_node(
            pipeline_id,
            {
                "parent_id": dataset_node["id"],
                "kind": "analysis",
                "name": task,
                "metadata": {"task_name": task},
                "request": request_defaults,
            },
        )
        next_snapshot = service.get_pipeline(pipeline_id)
        next_session = {"pipeline_id": pipeline_id, "selected_node_id": analysis_node["id"]}
        return next_session, next_snapshot, f"Analysis node added: {task}"

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Output("pipeline-store", "data", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("btn-add-join-node", "n_clicks"),
        Input("btn-add-ema-node", "n_clicks"),
        Input("btn-add-sma-node", "n_clicks"),
        Input("btn-add-transform-node", "n_clicks"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def on_add_utility_node(
        n_join: int,
        n_ema: int,
        n_sma: int,
        n_transform: int,
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
    ):
        if not session or not snapshot:
            return no_update, no_update, no_update
        trig = ctx.triggered_id
        clicks_by_trigger = {
            "btn-add-join-node": int(n_join or 0),
            "btn-add-ema-node": int(n_ema or 0),
            "btn-add-sma-node": int(n_sma or 0),
            "btn-add-transform-node": int(n_transform or 0),
        }
        if str(trig) not in clicks_by_trigger or clicks_by_trigger[str(trig)] <= 0:
            return no_update, no_update, no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        selected_node_id = str(session.get("selected_node_id") or "")
        node = _selected_node(snapshot, session)
        parent_id: str | None = None
        if not pipeline_id:
            return no_update, no_update, "WARN: No active pipeline"
        if not node:
            # When a virtual category node is selected, attach utility to latest analysis/utility node.
            if selected_node_id.startswith("virtual:utilities:"):
                analysis_id = selected_node_id.split(":", 2)[2]
                nodes = snapshot.get("nodes", {})
                if not isinstance(nodes, dict):
                    return no_update, no_update, "WARN: Invalid pipeline snapshot"
                utility_candidates = [
                    n for n in nodes.values()
                    if isinstance(n, dict) and n.get("kind") == "utility" and _ancestor_analysis_id(nodes, str(n.get("id"))) == analysis_id
                ]
                if utility_candidates:
                    utility_candidates.sort(key=lambda n: str(n.get("updated_at", "")))
                    parent_id = str(utility_candidates[-1].get("id"))
                else:
                    parent_id = analysis_id
            elif selected_node_id.startswith("virtual:"):
                parent_id = _latest_node_id(snapshot, ("utility", "analysis"))
                if not parent_id:
                    return no_update, no_update, "WARN: Add and apply an analysis node first"
            if parent_id:
                nodes = snapshot.get("nodes", {})
                node = nodes.get(parent_id) if isinstance(nodes, dict) else None
            if not node:
                return no_update, no_update, "WARN: Select a parent node first"
        else:
            parent_id = str(node.get("id"))

        util_by_trigger = {
            "btn-add-join-node": "join_tables",
            "btn-add-ema-node": "denoise_ema",
            "btn-add-sma-node": "denoise_sma",
            "btn-add-transform-node": "column_transform",
        }
        util_name = canonical_utility_name(util_by_trigger.get(str(trig), ""))
        if not util_name:
            return no_update, no_update, no_update
        source_art = _find_source_artifact(snapshot, str(parent_id or node.get("id") or ""), {})
        source_rows = _artifact_rows(source_art if isinstance(source_art, dict) else None)
        request = default_utility_request(util_name, columns=infer_columns(source_rows), numeric_columns=infer_numeric_columns(source_rows))
        catalog = _catalog_payload(service)
        utility_specs = _utility_specs_map(catalog)
        label = str(utility_specs.get(util_name, {}).get("label") or util_name)

        util_node = service.add_node(
            pipeline_id,
            {
                "parent_id": parent_id or node["id"],
                "kind": "utility",
                "name": util_name,
                "metadata": {"utility_name": util_name},
                "request": request,
            },
        )
        next_snapshot = service.get_pipeline(pipeline_id)
        next_session = {"pipeline_id": pipeline_id, "selected_node_id": util_node["id"]}
        return next_session, next_snapshot, f"Utility added: {label}"

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Output("pipeline-store", "data", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("btn-add-visualization-node", "n_clicks"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        State("viz-type", "value"),
        State("viz-x-col", "value"),
        State("viz-y-col", "value"),
        State("viz-z-col", "value"),
        State("viz-color-col", "value"),
        State("viz-group-col", "value"),
        State("viz-group-agg", "value"),
        prevent_initial_call=True,
    )
    def on_add_visualization_node(
        n_clicks: int,
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
        viz_type: str | None,
        x_col: str | None,
        y_col: str | None,
        z_col: str | None,
        color_col: str | None,
        group_col: str | None,
        group_agg: str | None,
    ):
        if not n_clicks or not session or not snapshot:
            return no_update, no_update, no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        if not pipeline_id:
            return no_update, no_update, "WARN: No active pipeline"

        selected_node_id = str(session.get("selected_node_id") or "")
        nodes = snapshot.get("nodes", {})
        if not isinstance(nodes, dict):
            return no_update, no_update, "WARN: Invalid pipeline snapshot"
        analysis_id = None
        if selected_node_id.startswith("virtual:visualization:"):
            analysis_id = selected_node_id.split(":", 2)[2]
        elif selected_node_id in nodes:
            analysis_id = _ancestor_analysis_id(nodes, selected_node_id)

        parent_id = None
        if analysis_id:
            parent_id = analysis_id
        if not parent_id:
            parent_id = _latest_node_id(snapshot, ("analysis",))
        if not parent_id:
            return no_update, no_update, "WARN: Add analysis/utility first"

        node = service.add_node(
            pipeline_id,
            {
                "parent_id": parent_id,
                "kind": "visualization",
                "name": str(viz_type or "plot2d"),
                "metadata": {"visualization_type": str(viz_type or "plot2d")},
                "request": {
                    "visualization_type": str(viz_type or "plot2d"),
                    "x_col": str(x_col or ""),
                    "y_col": str(y_col or ""),
                    "z_col": str(z_col or ""),
                    "plot_title": "",
                    "x_title": "",
                    "y_title": "",
                    "z_title": "",
                    "color_col": str(color_col or ""),
                    "group_col": str(group_col or ""),
                    "group_agg": _normalize_group_agg(group_agg),
                    "row_filters": [],
                    "line_color": "blue",
                    "line_color_rgb": "",
                    "line_width": 2.0,
                    "table_filter_col": "",
                    "table_filter_value": "",
                    "table_max_rows": 200,
                    "font_size": 12,
                    "marker_size": 0 if str(viz_type or "plot2d").lower() == "plot2d" else 6,
                    "theme": "plotly_white",
                    "axis_title_size": 13,
                    "grid_on": True,
                    "axis_box_on": True,
                    "axis_box_width": 1.5,
                    "log_scale": "none",
                    "tick_spacing_x": "",
                    "tick_spacing_y": "",
                    "show_markers": False,
                    "show_legend": True,
                    "legend_position": "top-right",
                    "trace_styles": {},
                },
            },
        )
        next_snapshot = service.get_pipeline(pipeline_id)
        return {"pipeline_id": pipeline_id, "selected_node_id": node["id"]}, next_snapshot, "Presentation added"

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Output("pipeline-store", "data", allow_duplicate=True),
        Output("result-store", "data", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("btn-delete-node", "n_clicks"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def on_delete_node(
        n_clicks: int,
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
    ):
        if not n_clicks or not session or not snapshot:
            return no_update, no_update, no_update, no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        node = _selected_node(snapshot, session)
        if not pipeline_id or not node:
            return no_update, no_update, no_update, "WARN: Select a node first"
        kind = str(node.get("kind") or "")
        if kind not in {"utility", "visualization"}:
            return no_update, no_update, no_update, "WARN: Delete is supported for utilities/presentations only"

        node_id = str(node.get("id"))
        nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
        analysis_id = _ancestor_analysis_id(nodes, node_id) if isinstance(nodes, dict) else None
        try:
            deleted = service.delete_node(pipeline_id, node_id)
        except Exception as exc:
            return no_update, no_update, no_update, f"ERROR: Delete failed: {exc}"
        deleted_node_ids = deleted.get("deleted_node_ids", []) if isinstance(deleted, dict) else []
        if isinstance(deleted_node_ids, list):
            for nid in deleted_node_ids:
                _NODE_PIPELINE_CACHE.pop(str(nid or "").strip(), None)
        deleted_artifact_ids = deleted.get("deleted_artifact_ids", []) if isinstance(deleted, dict) else []
        if isinstance(deleted_artifact_ids, list):
            for aid in deleted_artifact_ids:
                aid_txt = str(aid or "").strip()
                if not aid_txt:
                    continue
                _ARTIFACT_OBJ_CACHE.pop(aid_txt, None)
                _ARTIFACT_ROWS_CACHE.pop(aid_txt, None)
                purge_keys = [k for k in _PLOT_ROWS_CACHE.keys() if str(k).startswith(f"{aid_txt}|")]
                for key in purge_keys:
                    _PLOT_ROWS_CACHE.pop(key, None)

        next_snapshot = service.get_pipeline(pipeline_id)
        next_store = _result_cache_from_snapshot(next_snapshot)
        if analysis_id and kind == "utility":
            next_selected = f"virtual:utilities:{analysis_id}"
        elif analysis_id and kind == "visualization":
            next_selected = f"virtual:visualization:{analysis_id}"
        else:
            next_selected = "virtual:dataset"
        next_session = {"pipeline_id": pipeline_id, "selected_node_id": next_selected}
        return next_session, next_snapshot, next_store, "Node deleted"

    @app.callback(
        Output("status-banner", "className"),
        Input("status-banner", "children"),
    )
    def status_banner_class(message: str | None):
        text = str(message or "")
        if text.startswith("ERROR:"):
            return "rk-badge-error"
        if text.startswith("WARN:"):
            return "rk-badge-warn"
        return "rk-badge"

    @app.callback(
        Output("pipeline-browser-tree", "children"),
        Input("pipeline-store", "data"),
        Input("session-store", "data"),
    )
    def render_pipeline_nodes(snapshot: dict[str, Any] | None, session: dict[str, Any] | None):
        if not snapshot:
            return [html.Div("No nodes yet.", className="rk-tree-empty")]
        selected = (session or {}).get("selected_node_id")
        return _render_pipeline_tree(snapshot, selected)

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input({"type": "pipeline-node-btn", "node_id": ALL}, "n_clicks"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def on_select_node(_: list[int], session: dict[str, Any] | None):
        if session is None or "pipeline_id" not in session:
            return no_update
        if not ctx.triggered:
            return no_update
        triggered_value = ctx.triggered[0].get("value")
        if isinstance(triggered_value, (int, float)) and triggered_value <= 0:
            return no_update
        trig = ctx.triggered_id
        if not isinstance(trig, dict):
            return no_update
        node_id = trig.get("node_id")
        if not node_id:
            return no_update
        current_id = str(session.get("selected_node_id") or "")
        if str(node_id) == current_id:
            return no_update
        # Keep long-running execution stable: do not switch into a currently-running task node.
        try:
            live = service.get_pipeline(str(session["pipeline_id"]))
            live_nodes = live.get("nodes", {}) if isinstance(live, dict) else {}
            live_node = live_nodes.get(str(node_id)) if isinstance(live_nodes, dict) else None
            if isinstance(live_node, dict):
                is_running = str(live_node.get("status", "")).lower() == "running"
                if is_running and str(live_node.get("kind", "")) in {"analysis", "utility"}:
                    return no_update
        except Exception:
            pass
        return {"pipeline_id": session["pipeline_id"], "selected_node_id": node_id}

    @app.callback(
        Output("result-store", "data", allow_duplicate=True),
        Input("session-store", "data"),
        State("result-store", "data"),
        prevent_initial_call=True,
    )
    def sync_selected_node_result(session: dict[str, Any] | None, result_store: dict[str, Any] | None):
        if not session or "pipeline_id" not in session:
            return no_update
        pipeline_id = str(session["pipeline_id"])
        node_id = session.get("selected_node_id")
        if not node_id:
            return no_update
        if str(node_id).startswith("virtual:"):
            return no_update
        cache = dict(result_store or {})
        if node_id in cache:
            return no_update
        try:
            result = service.get_node_result(pipeline_id, str(node_id))
            if result and isinstance(result, dict) and "id" in result:
                cache[str(node_id)] = result
                return cache
        except Exception:
            return no_update
        return no_update

    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("btn-save-analysis-request-json", "n_clicks"),
        State("analysis-request-json", "value"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def on_save_analysis_request_json(
        n_clicks: int,
        raw_request: str | None,
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
    ):
        if not n_clicks or not session or not snapshot:
            return no_update, no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        node = _selected_node(snapshot, session)
        if not pipeline_id or not node or str(node.get("kind")) != "analysis" or str(node.get("name", "")).lower() == "msd":
            return no_update, no_update

        try:
            parsed = json.loads(str(raw_request or "").strip() or "{}")
        except Exception as exc:
            return no_update, f"WARN: Invalid analysis request JSON: {exc}"
        if not isinstance(parsed, dict):
            return no_update, "WARN: Analysis request must be a JSON object"
        request_payload = dict(parsed)
        old_req = node.get("request", {}) if isinstance(node.get("request"), dict) else {}
        if old_req == request_payload:
            return no_update, no_update
        service.update_node(pipeline_id, str(node["id"]), {"request": request_payload})
        return service.get_pipeline(pipeline_id), "Analysis parameters saved"

    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("btn-save-utility-request-json", "n_clicks"),
        State("utility-request-json", "value"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def on_save_utility_request_json(
        n_clicks: int,
        raw_request: str | None,
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
    ):
        if not n_clicks or not session or not snapshot:
            return no_update, no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        node = _selected_node(snapshot, session)
        if not pipeline_id or not node or str(node.get("kind")) != "utility":
            return no_update, no_update

        util_name = canonical_utility_name(str(node.get("metadata", {}).get("utility_name") or node.get("name") or ""))
        source_art = _find_source_artifact(snapshot, str(node.get("id") or ""), {})
        source_rows = _artifact_rows(source_art if isinstance(source_art, dict) else None)
        defaults = default_utility_request(util_name, columns=infer_columns(source_rows), numeric_columns=infer_numeric_columns(source_rows))
        try:
            parsed = json.loads(str(raw_request or "").strip() or "{}")
        except Exception as exc:
            return no_update, f"WARN: Invalid utility request JSON: {exc}"
        if not isinstance(parsed, dict):
            return no_update, "WARN: Utility request must be a JSON object"
        request_payload = dict(defaults)
        request_payload.update(parsed)
        old_req = node.get("request", {}) if isinstance(node.get("request"), dict) else {}
        if old_req == request_payload:
            return no_update, no_update
        service.update_node(pipeline_id, str(node["id"]), {"request": request_payload})
        return service.get_pipeline(pipeline_id), "Utility parameters saved"

    @app.callback(
        Output("util-join-keys", "options"),
        Output("util-join-keys", "value"),
        Input("util-join-right-source", "value"),
        State("util-join-keys", "value"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        State("result-store", "data"),
        prevent_initial_call=True,
    )
    def update_join_key_controls(
        right_source_id: str | None,
        current_keys: list[str] | None,
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
        result_store: dict[str, Any] | None,
    ):
        if not session or not snapshot:
            return no_update, no_update
        node = _selected_node(snapshot, session)
        if not node or str(node.get("kind")) != "utility":
            return no_update, no_update
        util_name = canonical_utility_name(str(node.get("metadata", {}).get("utility_name") or node.get("name") or ""))
        if util_name != "join_tables":
            return no_update, no_update

        left_source_id = str(node.get("parent_id") or "")
        left_source_art = _find_source_artifact(snapshot, left_source_id, result_store)
        right_source_art = _find_source_artifact(snapshot, str(right_source_id or ""), result_store)
        left_rows = _artifact_rows(left_source_art if isinstance(left_source_art, dict) else None)
        right_rows = _artifact_rows(right_source_art if isinstance(right_source_art, dict) else None)
        keys = _join_key_candidates(left_rows, right_rows)
        options = [{"label": key, "value": key} for key in keys]
        selected = [key for key in _normalize_join_keys(current_keys) if key in set(keys)]
        if not selected:
            selected = keys[: min(3, len(keys))]
        return options, selected

    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Input("util-join-right-source", "value"),
        Input("util-join-keys", "value"),
        Input("util-join-how", "value"),
        Input("util-join-left-suffix", "value"),
        Input("util-join-right-suffix", "value"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def on_live_join_utility_params(
        right_source_id: str | None,
        join_keys: list[str] | None,
        join_how: str | None,
        left_suffix: str | None,
        right_suffix: str | None,
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
    ):
        if not session or not snapshot:
            return no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        node = _selected_node(snapshot, session)
        if not pipeline_id or not node or str(node.get("kind")) != "utility":
            return no_update
        util_name = canonical_utility_name(str(node.get("metadata", {}).get("utility_name") or node.get("name") or ""))
        if util_name != "join_tables":
            return no_update

        old_req = node.get("request", {}) if isinstance(node.get("request"), dict) else {}
        request_payload = dict(old_req)
        request_payload["right_source_node_id"] = str(right_source_id or "").strip()
        request_payload["keys"] = _normalize_join_keys(join_keys)
        request_payload["how"] = str(join_how or "inner").strip().lower() or "inner"
        request_payload["left_suffix"] = str(left_suffix or "")
        request_payload["right_suffix"] = str(right_suffix or "_right")

        metadata = dict(node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {})
        source_node_ids = [str(node.get("parent_id") or "").strip(), str(right_source_id or "").strip()]
        metadata["source_node_ids"] = [value for value in source_node_ids if value]

        if request_payload == old_req and metadata == (node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {}):
            return no_update
        service.update_node(
            pipeline_id,
            str(node["id"]),
            {"request": request_payload, "metadata": metadata},
        )
        next_snapshot = _snapshot_with_node_update(
            snapshot,
            str(node["id"]),
            request=request_payload,
            metadata=metadata,
        )
        return next_snapshot if isinstance(next_snapshot, dict) else service.get_pipeline(pipeline_id)

    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Input({"type": "utility-field", "name": ALL}, "value"),
        State({"type": "utility-field", "name": ALL}, "id"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def on_live_utility_params(
        field_values: list[Any] | None,
        field_ids: list[dict[str, Any]] | None,
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
    ):
        if not session or not snapshot or not isinstance(field_ids, list) or not field_ids:
            return no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        node = _selected_node(snapshot, session)
        if not pipeline_id or not node or str(node.get("kind")) != "utility":
            return no_update

        util_name = canonical_utility_name(str(node.get("metadata", {}).get("utility_name") or node.get("name") or ""))
        if util_name == "join_tables":
            return no_update

        catalog = _catalog_payload(service)
        utility_spec = _utility_specs_map(catalog).get(util_name, {})
        spec_fields = utility_spec.get("fields", []) if isinstance(utility_spec, dict) else []
        if not isinstance(spec_fields, list) or not spec_fields:
            return no_update
        fields_by_name: dict[str, dict[str, Any]] = {}
        for field in spec_fields:
            if not isinstance(field, dict):
                continue
            fname = str(field.get("name") or "").strip()
            if fname:
                fields_by_name[fname] = field

        source_art = _find_source_artifact(snapshot, str(node.get("id")), {})
        source_rows = _artifact_rows(source_art if isinstance(source_art, dict) else None)
        default_req = default_utility_request(
            util_name,
            columns=infer_columns(source_rows),
            numeric_columns=infer_numeric_columns(source_rows),
        )
        old_req = node.get("request", {}) if isinstance(node.get("request"), dict) else {}
        request_payload = dict(default_req)
        request_payload.update(old_req)
        changed = False

        values = field_values if isinstance(field_values, list) else []
        for field_id, raw_value in zip(field_ids, values):
            if not isinstance(field_id, dict):
                continue
            fname = str(field_id.get("name") or "").strip()
            if not fname:
                continue
            field = fields_by_name.get(fname)
            if not field:
                continue
            ftype = str(field.get("type") or "Any")
            coerced = _coerce_utility_field_value(raw_value, ftype)
            if request_payload.get(fname) != coerced:
                request_payload[fname] = coerced
                changed = True

        if not changed or request_payload == old_req:
            return no_update
        service.update_node(pipeline_id, str(node["id"]), {"request": request_payload})
        next_snapshot = _snapshot_with_node_update(snapshot, str(node["id"]), request=request_payload)
        return next_snapshot if isinstance(next_snapshot, dict) else service.get_pipeline(pipeline_id)

    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Output("result-store", "data", allow_duplicate=True),
        Output("execute-loading-proxy", "children", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("btn-apply-node", "n_clicks"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        State("result-store", "data"),
        State("viz-row-filters-store", "data"),
        prevent_initial_call=True,
    )
    def on_apply_node(
        n_clicks: int,
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
        result_store: dict[str, Any] | None,
        viz_row_filters_store: dict[str, Any] | None,
    ):
        _trace(f"[UI_TRACE] on_apply_node called n_clicks={n_clicks} has_session={bool(session)} has_snapshot={bool(snapshot)}")
        if not n_clicks or not session or not snapshot:
            return no_update, no_update, no_update, no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        node = _selected_node(snapshot, session)
        if not pipeline_id or not node:
            return no_update, no_update, no_update, "WARN: Select a node first"

        node_id = str(node.get("id") or "")
        if str(node.get("kind") or "") == "visualization" and isinstance(viz_row_filters_store, dict):
            store_node_id = str(viz_row_filters_store.get("node_id") or "")
            viz_row_filters_data = viz_row_filters_store.get("rows")
            req_old = node.get("request", {}) if isinstance(node.get("request"), dict) else {}
            if store_node_id == node_id and isinstance(viz_row_filters_data, list):
                row_filters_from_ui = _row_filters_from_raw(viz_row_filters_data)
                row_filters_old = _row_filters_from_raw(req_old.get("row_filters"))
                if row_filters_from_ui != row_filters_old:
                    req_new = dict(req_old)
                    req_new["row_filters"] = row_filters_from_ui
                    try:
                        service.update_node(pipeline_id, node_id, {"request": req_new})
                        node = dict(node)
                        node["request"] = req_new
                    except Exception:
                        pass
        _trace(f"[UI_TRACE] on_apply_node executing pipeline_id={pipeline_id} node_id={node_id}")
        logger.info("ui.on_apply_node click=%s pipeline_id=%s node_id=%s node_kind=%s", n_clicks, pipeline_id, node_id, node.get("kind"))

        try:
            run_result = service.apply_node(pipeline_id, node_id)
        except Exception as exc:
            _trace(f"[UI_TRACE] on_apply_node error node_id={node_id} error={exc}")
            logger.exception("ui.on_apply_node failed pipeline_id=%s node_id=%s error=%s", pipeline_id, node_id, exc)
            return no_update, no_update, html.Span(str(n_clicks), style={"display": "none"}), f"ERROR: Execute failed: {exc}"

        artifact = run_result.get("artifact") if isinstance(run_result, dict) else None
        next_store = dict(result_store or {})
        if isinstance(artifact, dict) and "id" in artifact:
            _prime_rows_cache_from_artifact(artifact)
            _artifact_cache_put(artifact)
            artifact_id = str(artifact.get("id") or "").strip()
            if artifact_id:
                next_store[node_id] = artifact_id
            payload = artifact.get("payload", {})
            rows = extract_tabular_rows(payload if isinstance(payload, dict) else None)
            _trace(
                f"[UI_TRACE] on_apply_node artifact_id={artifact.get('id')} payload_keys={sorted(payload.keys()) if isinstance(payload, dict) else []} rows={len(rows)} cols={list(rows[0].keys()) if rows else []}"
            )
            logger.info(
                "ui.on_apply_node artifact_id=%s payload_keys=%s rows=%s cols=%s",
                artifact.get("id"),
                sorted(payload.keys()) if isinstance(payload, dict) else [],
                len(rows),
                list(rows[0].keys()) if rows else [],
            )
            if str(node.get("kind")) == "utility":
                nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
                if isinstance(nodes, dict):
                    analysis_id = _ancestor_analysis_id(nodes, node_id)
                    if analysis_id and artifact_id:
                        next_store[str(analysis_id)] = artifact_id
        next_snapshot = service.get_pipeline(pipeline_id)
        if str(node.get("kind")) == "analysis" and isinstance(artifact, dict):
            # Create recommended visualization nodes under this analysis (once).
            children = next_snapshot.get("children", {})
            existing_vis = []
            if isinstance(children, dict):
                for child_id in children.get(str(node["id"]), []):
                    cn = next_snapshot.get("nodes", {}).get(str(child_id), {}) if isinstance(next_snapshot.get("nodes", {}), dict) else {}
                    if isinstance(cn, dict) and str(cn.get("kind")) == "visualization":
                        existing_vis.append(cn)
            if not existing_vis:
                recs = artifact.get("recommended_views", [])
                if isinstance(recs, list):
                    for rec in recs:
                        if not isinstance(rec, dict):
                            continue
                        spec = ensure_presentation_spec(rec)
                        req = spec_to_dash_request(spec or rec)
                        vtype = str(req.get("visualization_type") or "plot2d").lower()
                        name = str((spec.label if spec else rec.get("label")) or vtype)
                        service.add_node(
                            pipeline_id,
                            {
                                "parent_id": str(node["id"]),
                                "kind": "visualization",
                                "name": name,
                                "metadata": {
                                    "visualization_type": vtype,
                                    "auto_recommended": True,
                                    "presentation_spec": rec,
                                },
                                "request": req,
                            },
                        )
                next_snapshot = service.get_pipeline(pipeline_id)
        try:
            if isinstance(artifact, dict):
                kind = str(node.get("kind") or "")
                prime_analysis_id = ""
                if kind == "analysis":
                    prime_analysis_id = str(node_id)
                elif kind == "utility":
                    nodes_latest = next_snapshot.get("nodes", {}) if isinstance(next_snapshot, dict) else {}
                    if isinstance(nodes_latest, dict):
                        prime_analysis_id = str(_ancestor_analysis_id(nodes_latest, node_id) or "")
                if prime_analysis_id:
                    primed = _precompute_plot2d_cache_for_analysis(
                        snapshot=next_snapshot,
                        analysis_id=prime_analysis_id,
                        result_store=next_store,
                    )
                    if primed > 0:
                        logger.info(
                            "ui.precompute_plot2d_cache pipeline_id=%s analysis_id=%s primed=%s",
                            pipeline_id,
                            prime_analysis_id,
                            primed,
                        )
        except Exception as exc:
            logger.debug(
                "ui.precompute_plot2d_cache skipped pipeline_id=%s node_id=%s error=%s",
                pipeline_id,
                node_id,
                exc,
            )
        return next_snapshot, next_store, html.Span(str(n_clicks), style={"display": "none"}), "Node executed"

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

    @app.callback(
        Output("config-store", "data", allow_duplicate=True),
        Input("viz-type", "value"),
        State("session-store", "data"),
        State("config-store", "data"),
        prevent_initial_call=True,
    )
    def sync_virtual_viz_type(
        viz_type: str | None,
        session: dict[str, Any] | None,
        config: dict[str, Any] | None,
    ):
        selected_id = str((session or {}).get("selected_node_id") or "")
        if not selected_id.startswith("virtual:visualization"):
            return no_update
        cfg = dict(config or {})
        cfg["draft_viz_type"] = str(viz_type or "plot2d")
        return cfg

    @app.callback(
        Output("viz-color-name-wrap", "style"),
        Input("viz-line-color-rgb", "value"),
        State("viz-type", "value"),
        prevent_initial_call=False,
    )
    def toggle_viz_color_name(rgb_value: str | None, viz_type: str | None):
        if str(viz_type or "") not in {"plot2d", "histogram", "scatter3d"}:
            return {"display": "none"}
        if rgb_value and str(rgb_value).strip():
            return {"display": "none"}
        return {"display": "grid"}

    @app.callback(
        Output("viz-row-filters", "data", allow_duplicate=True),
        Input("btn-viz-add-row-filter", "n_clicks"),
        State("viz-row-filters", "data"),
        prevent_initial_call=True,
    )
    def on_add_viz_row_filter(n_clicks: int, rows_data: list[dict[str, Any]] | None):
        if not n_clicks:
            return no_update
        rows = _row_filter_table_data(rows_data)
        rows.append({"column": "", "op": "==", "value": ""})
        return rows

    @app.callback(
        Output("viz-row-filters-store", "data", allow_duplicate=True),
        Input("viz-row-filters", "data"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def sync_viz_row_filters_store(
        rows_data: list[dict[str, Any]] | None,
        session: dict[str, Any] | None,
    ):
        node_id = str((session or {}).get("selected_node_id") or "")
        return {"node_id": node_id, "rows": _row_filter_table_data(rows_data)}

    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("btn-save-viz-params", "n_clicks"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        State("viz-type", "value"),
        State("viz-x-col", "value"),
        State("viz-y-col", "value"),
        State("viz-z-col", "value"),
        State("viz-color-col", "value"),
        State("viz-group-col", "value"),
        State("viz-group-agg", "value"),
        State("viz-row-filters", "data"),
        State("viz-use-plot-title", "value"),
        State("viz-plot-title", "value"),
        State("viz-x-title", "value"),
        State("viz-y-title", "value"),
        State("viz-z-title", "value"),
        State("viz-line-color-name", "value"),
        State("viz-line-color-rgb", "value"),
        State("viz-line-width", "value"),
        State("viz-font-size", "value"),
        State("viz-marker-size", "value"),
        State("viz-theme", "value"),
        State("viz-axis-title-size", "value"),
        State("viz-grid-on", "value"),
        State("viz-axis-box-on", "value"),
        State("viz-axis-box-width", "value"),
        State("viz-log-scale", "value"),
        State("viz-tick-spacing-x", "value"),
        State("viz-tick-spacing-y", "value"),
        State("viz-show-markers", "value"),
        State("viz-show-legend", "value"),
        State("viz-legend-position", "value"),
        State("viz-table-filter-col", "value"),
        State("viz-table-filter-value", "value"),
        State("viz-table-max-rows", "value"),
        prevent_initial_call=True,
    )
    def on_save_viz_params(
        n_clicks: int,
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
        viz_type: str | None,
        x_col: str | None,
        y_col: str | None,
        z_col: str | None,
        color_col: str | None,
        group_col: str | None,
        group_agg: str | None,
        row_filters_data: list[dict[str, Any]] | None,
        use_plot_title_values: list[str] | None,
        plot_title: str | None,
        x_title: str | None,
        y_title: str | None,
        z_title: str | None,
        line_color_name: str | None,
        line_color_rgb: str | None,
        line_width: float | None,
        font_size: float | None,
        marker_size: float | None,
        theme: str | None,
        axis_title_size: float | None,
        grid_on_values: list[str] | None,
        axis_box_on_values: list[str] | None,
        axis_box_width: float | None,
        log_scale: str | None,
        tick_spacing_x: float | None,
        tick_spacing_y: float | None,
        show_markers_values: list[str] | None,
        show_legend_values: list[str] | None,
        legend_position: str | None,
        table_filter_col: str | None,
        table_filter_value: str | None,
        table_max_rows: int | None,
    ):
        if not n_clicks or not session or not snapshot:
            return no_update, no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        node = _selected_node(snapshot, session)
        if not pipeline_id or not node or node.get("kind") != "visualization":
            return no_update, "WARN: Select a visualization node"
        req_old = node.get("request", {}) if isinstance(node.get("request"), dict) else {}
        payload = {
            "visualization_type": str(viz_type or "plot2d"),
            "x_col": str(x_col or ""),
            "y_col": str(y_col or ""),
            "z_col": str(z_col or ""),
            "use_plot_title": bool("on" in (use_plot_title_values or [])),
            "plot_title": str(plot_title or ""),
            "x_title": str(x_title or ""),
            "y_title": str(y_title or ""),
            "z_title": str(z_title or ""),
            "color_col": str(color_col or ""),
            "group_col": str(group_col or ""),
            "group_agg": _normalize_group_agg(group_agg),
            "row_filters": _row_filter_table_data(row_filters_data),
            "line_color": str(line_color_name or "blue"),
            "line_color_rgb": str(line_color_rgb or ""),
            "line_width": float(line_width if line_width is not None else 2.0),
            "font_size": float(font_size if font_size is not None else 12.0),
            "marker_size": float(marker_size if marker_size is not None else 0.0),
            "theme": _safe_theme(theme or "plotly_white"),
            "axis_title_size": float(axis_title_size if axis_title_size is not None else 13.0),
            "grid_on": bool("on" in (grid_on_values or [])),
            "axis_box_on": bool("on" in (axis_box_on_values or [])),
            "axis_box_width": float(axis_box_width if axis_box_width is not None else 1.5),
            "log_scale": str(log_scale or "none"),
            "tick_spacing_x": "" if tick_spacing_x is None else float(tick_spacing_x),
            "tick_spacing_y": "" if tick_spacing_y is None else float(tick_spacing_y),
            "show_markers": bool("on" in (show_markers_values or [])),
            "show_legend": bool("on" in (show_legend_values or [])),
            "legend_position": str(legend_position or "top-right"),
            "table_filter_col": str(table_filter_col or ""),
            "table_filter_value": str(table_filter_value or ""),
            "table_max_rows": int(table_max_rows) if table_max_rows is not None else 200,
            "trace_styles": _trace_styles_map(req_old.get("trace_styles")),
        }
        service.update_node(pipeline_id, str(node["id"]), {"request": payload, "metadata": {"visualization_type": payload["visualization_type"]}})
        next_snapshot = service.get_pipeline(pipeline_id)
        return next_snapshot, "Presentation parameters saved"

    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Input("viz-type", "value"),
        Input("viz-x-col", "value"),
        Input("viz-y-col", "value"),
        Input("viz-z-col", "value"),
        Input("viz-color-col", "value"),
        Input("viz-group-col", "value"),
        Input("viz-group-agg", "value"),
        Input("viz-row-filters", "data"),
        Input("viz-use-plot-title", "value"),
        Input("viz-plot-title", "value"),
        Input("viz-x-title", "value"),
        Input("viz-y-title", "value"),
        Input("viz-z-title", "value"),
        Input("viz-line-color-name", "value"),
        Input("viz-line-color-rgb", "value"),
        Input("viz-line-width", "value"),
        Input("viz-font-size", "value"),
        Input("viz-marker-size", "value"),
        Input("viz-theme", "value"),
        Input("viz-axis-title-size", "value"),
        Input("viz-grid-on", "value"),
        Input("viz-axis-box-on", "value"),
        Input("viz-axis-box-width", "value"),
        Input("viz-log-scale", "value"),
        Input("viz-tick-spacing-x", "value"),
        Input("viz-tick-spacing-y", "value"),
        Input("viz-show-markers", "value"),
        Input("viz-show-legend", "value"),
        Input("viz-legend-position", "value"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def on_live_plot2d_params(
        viz_type: str | None,
        x_col: str | None,
        y_col: str | None,
        z_col: str | None,
        color_col: str | None,
        group_col: str | None,
        group_agg: str | None,
        row_filters_data: list[dict[str, Any]] | None,
        use_plot_title_values: list[str] | None,
        plot_title: str | None,
        x_title: str | None,
        y_title: str | None,
        z_title: str | None,
        line_color_name: str | None,
        line_color_rgb: str | None,
        line_width: float | None,
        font_size: float | None,
        marker_size: float | None,
        theme: str | None,
        axis_title_size: float | None,
        grid_on_values: list[str] | None,
        axis_box_on_values: list[str] | None,
        axis_box_width: float | None,
        log_scale: str | None,
        tick_spacing_x: float | None,
        tick_spacing_y: float | None,
        show_markers_values: list[str] | None,
        show_legend_values: list[str] | None,
        legend_position: str | None,
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
    ):
        if not session or not snapshot:
            return no_update
        use_type = str(viz_type or "").strip().lower()
        if use_type not in {"plot2d", "scatter3d", "histogram"}:
            return no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        node = _selected_node(snapshot, session)
        if not pipeline_id or not node or node.get("kind") != "visualization":
            return no_update
        req_old = node.get("request", {}) if isinstance(node.get("request"), dict) else {}
        payload = dict(req_old)
        payload.update(
            {
                "visualization_type": use_type,
                "x_col": str(x_col or payload.get("x_col") or ""),
                "y_col": str(y_col or payload.get("y_col") or ""),
                "z_col": str(z_col or payload.get("z_col") or ""),
                "use_plot_title": bool("on" in (use_plot_title_values or [])),
                "plot_title": str(plot_title or payload.get("plot_title") or ""),
                "x_title": str(x_title or payload.get("x_title") or ""),
                "y_title": str(y_title or payload.get("y_title") or ""),
                "z_title": str(z_title or payload.get("z_title") or ""),
                "color_col": str(color_col or payload.get("color_col") or ""),
                "group_col": str(group_col or payload.get("group_col") or ""),
                "group_agg": _normalize_group_agg(group_agg if group_agg is not None else payload.get("group_agg")),
                "row_filters": _row_filter_table_data(row_filters_data),
                "line_color": str(line_color_name or payload.get("line_color") or "blue"),
                "line_color_rgb": str(line_color_rgb or payload.get("line_color_rgb") or ""),
                "line_width": float(line_width if line_width is not None else float(payload.get("line_width") or 2.0)),
                "font_size": float(font_size if font_size is not None else float(payload.get("font_size") or 12.0)),
                "marker_size": float(marker_size if marker_size is not None else float(payload.get("marker_size") or 0.0)),
                "theme": _safe_theme(theme or payload.get("theme") or "plotly_white"),
                "axis_title_size": float(axis_title_size if axis_title_size is not None else float(payload.get("axis_title_size") or 13.0)),
                "grid_on": bool("on" in (grid_on_values or [])),
                "axis_box_on": bool("on" in (axis_box_on_values or [])),
                "axis_box_width": float(axis_box_width if axis_box_width is not None else float(payload.get("axis_box_width") or 1.5)),
                "log_scale": str(log_scale or payload.get("log_scale") or "none"),
                "tick_spacing_x": "" if tick_spacing_x is None else float(tick_spacing_x),
                "tick_spacing_y": "" if tick_spacing_y is None else float(tick_spacing_y),
                "show_markers": bool("on" in (show_markers_values or [])),
                "show_legend": bool("on" in (show_legend_values or [])),
                "legend_position": str(legend_position or payload.get("legend_position") or "top-right"),
            }
        )
        if use_type == "plot2d":
            payload["marker_size"] = float(marker_size if marker_size is not None else float(payload.get("marker_size") or 0.0))
        elif use_type == "scatter3d":
            payload["line_width"] = float(payload.get("line_width") or 2.0)
            payload["marker_size"] = float(marker_size if marker_size is not None else float(payload.get("marker_size") or 6.0))
            payload["log_scale"] = "none"
            payload["tick_spacing_x"] = ""
            payload["tick_spacing_y"] = ""
            if not payload.get("legend_position"):
                payload["legend_position"] = "top-right"
        elif use_type == "histogram":
            payload["line_width"] = float(payload.get("line_width") or 2.0)
            payload["marker_size"] = 0.0
            if str(payload.get("log_scale") or "none") not in {"none", "y"}:
                payload["log_scale"] = "none"
            if not payload.get("legend_position"):
                payload["legend_position"] = "top-right"
        if payload == req_old:
            return no_update
        service.update_node(
            pipeline_id,
            str(node["id"]),
            {"request": payload, "metadata": {"visualization_type": payload["visualization_type"]}},
        )
        next_snapshot = _snapshot_with_node_update(
            snapshot,
            str(node["id"]),
            request=payload,
            metadata={"visualization_type": payload["visualization_type"]},
        )
        return next_snapshot if isinstance(next_snapshot, dict) else service.get_pipeline(pipeline_id)

    @app.callback(
        Output("selected-curve-store", "data"),
        Input("session-store", "data"),
        Input({"type": "plot-graph", "slot": ALL}, "clickData"),
        State({"type": "plot-graph", "slot": ALL}, "id"),
        prevent_initial_call=True,
    )
    def on_curve_click(
        session: dict[str, Any] | None,
        click_data_all: list[dict[str, Any] | None] | None,
        graph_ids: list[dict[str, Any]] | None,
    ):
        triggered = ctx.triggered_id
        if triggered == "session-store":
            return {}
        if not session or not isinstance(click_data_all, list):
            return no_update
        click_data: dict[str, Any] | None = None
        if isinstance(triggered, dict):
            trigger_slot = str(triggered.get("slot") or "")
            for gid, cdata in zip(graph_ids or [], click_data_all):
                if not isinstance(gid, dict):
                    continue
                if str(gid.get("slot") or "") == trigger_slot and isinstance(cdata, dict):
                    click_data = cdata
                    break
        if click_data is None:
            click_data = next((c for c in click_data_all if isinstance(c, dict)), None)
        if not isinstance(click_data, dict):
            return no_update
        points = click_data.get("points")
        if not isinstance(points, list) or not points:
            return no_update
        point = points[0] if isinstance(points[0], dict) else {}
        try:
            curve_index = int(point.get("curveNumber"))
        except Exception:
            return no_update
        if curve_index < 0:
            return no_update
        data_obj = point.get("data") if isinstance(point.get("data"), dict) else {}
        full_obj = point.get("fullData") if isinstance(point.get("fullData"), dict) else {}
        trace_name = str(data_obj.get("name") or full_obj.get("name") or "").strip()
        trace_key = trace_name or f"curve_{curve_index}"
        trace_label = trace_name or f"curve {curve_index}"
        return {
            "node_id": str(session.get("selected_node_id") or ""),
            "curve_index": curve_index,
            "trace_key": trace_key,
            "trace_label": trace_label,
        }

    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Input("viz-curve-color-name", "value"),
        Input("viz-curve-color-rgb", "value"),
        Input("viz-curve-line-width", "value"),
        Input("viz-curve-marker-size", "value"),
        Input("viz-curve-show-markers", "value"),
        State("viz-selected-trace-key", "value"),
        State("selected-curve-store", "data"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def on_live_curve_params(
        curve_color_name: str | None,
        curve_color_rgb: str | None,
        curve_line_width: float | None,
        curve_marker_size: float | None,
        curve_show_markers_values: list[str] | None,
        selected_trace_key_value: str | None,
        selected_curve_state: dict[str, Any] | None,
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
    ):
        if not session or not snapshot:
            return no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        node = _selected_node(snapshot, session)
        if not pipeline_id or not node or node.get("kind") != "visualization":
            return no_update
        req_old = node.get("request", {}) if isinstance(node.get("request"), dict) else {}
        if str(req_old.get("visualization_type") or "plot2d").lower() != "plot2d":
            return no_update

        selected_curve = dict(selected_curve_state or {})
        trace_key = str(selected_trace_key_value or selected_curve.get("trace_key") or "").strip()
        if not trace_key:
            return no_update

        trace_styles = _trace_styles_map(req_old.get("trace_styles"))
        style_old = trace_styles.get(trace_key, {}) if isinstance(trace_styles.get(trace_key), dict) else {}
        style_new = dict(style_old)
        style_new.update(
            {
                "line_color": str(curve_color_name or "blue"),
                "line_color_rgb": str(curve_color_rgb or ""),
                "line_width": float(curve_line_width if curve_line_width is not None else _parse_float(req_old.get("line_width"), 2.0) or 2.0),
                "marker_size": float(curve_marker_size if curve_marker_size is not None else _parse_float(req_old.get("marker_size"), 6.0) or 6.0),
                "show_markers": bool("on" in (curve_show_markers_values or [])),
            }
        )

        if style_new == style_old:
            return no_update
        trace_styles[trace_key] = style_new
        payload = dict(req_old)
        payload["trace_styles"] = trace_styles
        service.update_node(
            pipeline_id,
            str(node["id"]),
            {"request": payload, "metadata": {"visualization_type": payload.get("visualization_type", "plot2d")}},
        )
        next_snapshot = _snapshot_with_node_update(
            snapshot,
            str(node["id"]),
            request=payload,
            metadata={"visualization_type": payload.get("visualization_type", "plot2d")},
        )
        return next_snapshot if isinstance(next_snapshot, dict) else service.get_pipeline(pipeline_id)

    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Input("viz-type", "value"),
        Input("viz-table-max-rows", "value"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def on_live_table_params(
        viz_type: str | None,
        table_max_rows: int | None,
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
    ):
        if not session or not snapshot:
            return no_update
        if str(viz_type or "") != "table":
            return no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        node = _selected_node(snapshot, session)
        if not pipeline_id or not node or node.get("kind") != "visualization":
            return no_update
        req_old = node.get("request", {}) if isinstance(node.get("request"), dict) else {}
        payload = dict(req_old)
        payload["visualization_type"] = "table"
        payload["table_filter_col"] = ""
        payload["table_filter_value"] = ""
        payload["table_max_rows"] = int(table_max_rows) if table_max_rows is not None else int(req_old.get("table_max_rows") or 200)
        if payload == req_old:
            return no_update
        service.update_node(
            pipeline_id,
            str(node["id"]),
            {"request": payload, "metadata": {"visualization_type": payload["visualization_type"]}},
        )
        next_snapshot = _snapshot_with_node_update(
            snapshot,
            str(node["id"]),
            request=payload,
            metadata={"visualization_type": payload["visualization_type"]},
        )
        return next_snapshot if isinstance(next_snapshot, dict) else service.get_pipeline(pipeline_id)

    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Input("viz-type", "value"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def on_visualization_type_change(
        viz_type: str | None,
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
    ):
        if not session or not snapshot:
            return no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        node = _selected_node(snapshot, session)
        if not pipeline_id or not node or node.get("kind") != "visualization":
            return no_update
        req_old = node.get("request", {}) if isinstance(node.get("request"), dict) else {}
        target_type = str(viz_type or req_old.get("visualization_type") or "plot2d")
        if str(req_old.get("visualization_type") or "plot2d") == target_type:
            return no_update
        payload = _viz_request_with_defaults(req_old, target_type)
        if target_type == "scatter3d":
            if _parse_float(payload.get("marker_size"), 0.0) in {0.0, None}:
                payload["marker_size"] = 6
        elif target_type == "plot2d":
            if payload.get("marker_size") is None:
                payload["marker_size"] = 0
        service.update_node(
            pipeline_id,
            str(node["id"]),
            {"request": payload, "metadata": {"visualization_type": payload["visualization_type"]}},
        )
        next_snapshot = _snapshot_with_node_update(
            snapshot,
            str(node["id"]),
            request=payload,
            metadata={"visualization_type": payload["visualization_type"]},
        )
        return next_snapshot if isinstance(next_snapshot, dict) else service.get_pipeline(pipeline_id)

    @app.callback(
        Output("input-dataset-path", "value"),
        Input("btn-browse-dataset", "n_clicks"),
        State("input-dataset-path", "value"),
        prevent_initial_call=True,
    )
    def on_browse_dataset(n_clicks: int, current_value: str | None):
        if not n_clicks:
            return no_update
        path = _browse_directory()
        if not path:
            return current_value or "."
        return path

    @app.callback(
        Output("input-workspace-dir", "value"),
        Output("input-workspace-dir", "disabled"),
        Input("input-default-workspace", "value"),
        Input("input-dataset-path", "value"),
        State("input-workspace-dir", "value"),
        prevent_initial_call=False,
    )
    def sync_workspace_dir_input(
        default_flags: list[str] | None,
        dataset_path: str | None,
        current_value: str | None,
    ):
        use_default = "default" in (default_flags or [])
        if use_default:
            return _default_workspace_dir_for_dataset(dataset_path), True
        return str(current_value or _default_workspace_dir_for_dataset(dataset_path)), False

    @app.callback(
        Output("engine-file-roles", "style"),
        Input("input-engine-name", "value"),
        prevent_initial_call=False,
    )
    def toggle_engine_file_roles(engine_name: str | None):
        eng = str(engine_name or "autodetect").lower()
        visible = eng != "autodetect"
        return {"display": "grid"} if visible else {"display": "none"}

    @app.callback(
        Output("dataset-info-content", "children"),
        Input("pipeline-store", "data"),
    )
    def render_dataset_info(snapshot: dict[str, Any] | None):
        if not snapshot:
            return "No dataset loaded."
        dataset_node = _latest_dataset_node(snapshot)
        if not dataset_node:
            return "No dataset loaded."
        meta = dataset_node.get("metadata", {})
        dataset = meta.get("dataset", {}) if isinstance(meta, dict) else {}
        frames = dataset.get("frames")
        if frames is None:
            frames = "unknown"
        return (
            f"Frames: {frames} | "
            f"Engine: {_engine_display_name(dataset.get('engine_override') or dataset.get('engine_detected') or 'unknown')}"
        )

    @app.callback(
        Output("btn-canvas-primary", "children"),
        Output("btn-canvas-secondary", "children"),
        Output("btn-canvas-primary", "style"),
        Output("btn-canvas-secondary", "style"),
        Input("session-store", "data"),
        Input("pipeline-store", "data"),
        prevent_initial_call=False,
    )
    def render_canvas_action_buttons(
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
    ):
        source_node = _resolve_canvas_source_node(snapshot, session, None)
        if not isinstance(source_node, dict):
            hidden = {"display": "none"}
            return "Save", "Save As", hidden, hidden
        kind = str(source_node.get("kind") or "")
        req = source_node.get("request", {}) if isinstance(source_node.get("request"), dict) else {}
        viz_type = str(req.get("visualization_type") or "plot2d").lower()
        mode = "table" if kind == "utility" or viz_type == "table" else "plot"
        visible = {"display": "block"}
        if mode == "table":
            return "Export", "Export As", visible, visible
        return "Save", "Save As", visible, visible

    @app.callback(
        Output("canvas-export-status", "children"),
        Output("status-banner", "children", allow_duplicate=True),
        Output("execute-loading-proxy", "children", allow_duplicate=True),
        Input("btn-canvas-primary", "n_clicks"),
        Input("btn-canvas-secondary", "n_clicks"),
        State("session-store", "data"),
        State("result-store", "data"),
        State("pipeline-store", "data"),
        State("config-store", "data"),
        State({"type": "plot-graph", "slot": ALL}, "figure"),
        State({"type": "plot-graph", "slot": ALL}, "id"),
        prevent_initial_call=True,
    )
    def on_canvas_save_export(
        n_primary: int,
        n_secondary: int,
        session: dict[str, Any] | None,
        result_store: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
        config: dict[str, Any] | None,
        graph_figures: list[dict[str, Any] | None] | None,
        graph_ids: list[dict[str, Any] | None] | None,
    ):
        trig = str(ctx.triggered_id or "")
        if trig not in {"btn-canvas-primary", "btn-canvas-secondary"}:
            return no_update, no_update, no_update
        if trig == "btn-canvas-primary" and int(n_primary or 0) <= 0:
            return no_update, no_update, no_update
        if trig == "btn-canvas-secondary" and int(n_secondary or 0) <= 0:
            return no_update, no_update, no_update

        mode, rows, fig = _build_canvas_payload(
            snapshot=snapshot,
            session=session,
            result_store=result_store,
            tab_node_id=None,
            x_col=None,
            y_col=None,
            group_col=None,
            hist_col=None,
            view3d_x=None,
            view3d_y=None,
            view3d_z=None,
            view3d_color=None,
            focus_atom=None,
        )
        spinner_tick = html.Span(str(int(n_primary or 0) + int(n_secondary or 0)), style={"display": "none"})
        if mode == "none":
            return "Nothing to save/export.", "Nothing to save/export.", spinner_tick

        source_node = _resolve_canvas_source_node(snapshot, session, None)
        nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
        source_id = str(source_node.get("id") or "") if isinstance(source_node, dict) else ""
        analysis_id = _ancestor_analysis_id(nodes, source_id) if isinstance(nodes, dict) and source_id else None
        workflow = "analysis"
        if analysis_id and isinstance(nodes, dict):
            anode = nodes.get(analysis_id)
            if isinstance(anode, dict):
                workflow = str(anode.get("metadata", {}).get("task_name") or anode.get("name") or "analysis")
        rel_save_dir = f"{workflow}/{analysis_id}" if analysis_id else workflow

        # Prefer the exact figure currently shown on canvas.
        if mode == "plot" and (not isinstance(fig, go.Figure)):
            chosen_fig: dict[str, Any] | None = None
            for gid, gfig in zip(graph_ids or [], graph_figures or []):
                if isinstance(gid, dict) and str(gid.get("slot") or "") == "canvas" and isinstance(gfig, dict):
                    chosen_fig = gfig
                    break
            if chosen_fig is None:
                for gfig in (graph_figures or []):
                    if isinstance(gfig, dict):
                        chosen_fig = gfig
                        break
            if isinstance(chosen_fig, dict):
                fig = go.Figure(chosen_fig)
        export_dir = _default_export_dir(snapshot, session, config)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            if mode == "table":
                if trig == "btn-canvas-primary":
                    path = export_dir / f"table_{stamp}.csv"
                    write_table(rows, path, "csv")
                    msg = f"saved to {rel_save_dir}"
                    return msg, msg, spinner_tick
                chosen = _browse_save_file(
                    title="Export table as...",
                    initial_dir=export_dir,
                    initial_name=f"table_{stamp}.csv",
                    filetypes=[("CSV", "*.csv"), ("Excel Workbook", "*.xlsx")],
                    default_ext=".csv",
                )
                if chosen is None:
                    return "Export canceled.", "Export canceled.", spinner_tick
                suffix = chosen.suffix.lower()
                fmt = "xlsx" if suffix == ".xlsx" else "csv"
                if suffix not in {".csv", ".xlsx"}:
                    chosen = chosen.with_suffix(".csv")
                    fmt = "csv"
                write_table(rows, chosen, fmt)
                msg = "saved in the requested directory"
                return msg, msg, spinner_tick

            if fig is None:
                return "No figure available to save.", "No figure available to save.", spinner_tick
            if trig == "btn-canvas-primary":
                path = export_dir / f"plot_{stamp}.png"
                write_figure(fig, path, "png")
                msg = f"saved to {rel_save_dir}"
                return msg, msg, spinner_tick
            chosen = _browse_save_file(
                title="Save plot as...",
                initial_dir=export_dir,
                initial_name=f"plot_{stamp}.png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpeg;*.jpg")],
                default_ext=".png",
            )
            if chosen is None:
                return "Save canceled.", "Save canceled.", spinner_tick
            suffix = chosen.suffix.lower()
            fmt = "jpeg" if suffix in {".jpeg", ".jpg"} else "png"
            if suffix not in {".png", ".jpeg", ".jpg"}:
                chosen = chosen.with_suffix(".png")
                fmt = "png"
            write_figure(fig, chosen, fmt)
            msg = "saved in the requested directory"
            return msg, msg, spinner_tick
        except Exception as exc:
            msg = f"Save/export failed: {exc}"
            return msg, msg, spinner_tick

    @app.callback(
        Output("canvas-content", "children"),
        Input("session-store", "data"),
        Input("result-store", "data"),
        Input("pipeline-store", "data"),
    )
    def render_result_views(
        session: dict[str, Any] | None,
        result_store: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
    ):
        if not session:
            return "No selected node."
        node_id = str(session.get("selected_node_id") or "")
        nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
        selected_node = nodes.get(node_id) if isinstance(nodes, dict) else None
        if isinstance(selected_node, dict) and str(selected_node.get("kind")) == "analysis":
            return "No presentation selected. Select a presentation node under this analysis."
        if str(node_id).startswith("virtual:visualization:"):
            # Selecting the virtual Presentation folder should not change the canvas view.
            return no_update

        source_node = _resolve_canvas_source_node(snapshot, session, None)
        if not isinstance(source_node, dict):
            return "No presentation selected."
        source_node_id = str(source_node.get("id") or "")
        if str(source_node.get("kind")) == "utility":
            artifact = _find_source_artifact(snapshot, source_node_id, result_store or {})
            rows = _artifact_rows(artifact if isinstance(artifact, dict) else None)
            if not rows:
                return "No result rows yet."
            return _build_result_table(rows, max_rows=200)
        presentation_spec = None
        if isinstance(source_node, dict):
            meta = source_node.get("metadata", {})
            if isinstance(meta, dict):
                presentation_spec = meta.get("presentation_spec")
        artifact = _find_source_artifact(snapshot, source_node_id, result_store or {})
        rows = _artifact_rows(artifact if isinstance(artifact, dict) else None)
        req = source_node.get("request", {}) if isinstance(source_node.get("request"), dict) else {}
        rows = _apply_row_filters(rows, req.get("row_filters"))
        viz_type = str(req.get("visualization_type") or "plot2d").lower()

        if viz_type == "table":
            if not rows:
                return "No result rows yet."
            table_max_rows = int(req.get("table_max_rows") or 200)
            return _build_result_table(rows, max_rows=table_max_rows)

        if viz_type == "plot2d":
            artifact_id = str((artifact or {}).get("id") if isinstance(artifact, dict) else "")
            fig = _build_plot2d_figure(
                rows=rows,
                artifact_id=artifact_id,
                presentation_spec=presentation_spec if isinstance(presentation_spec, dict) else None,
                req=req,
                x_col=None,
                y_col=None,
                group_col=None,
                x_range=None,
                max_points_total=_UI_PLOT2D_INITIAL_MAX_POINTS,
                cache_scope="initial",
            )
            if fig is None:
                return "No plottable data."
            graph_canvas = dcc.Graph(
                id={"type": "plot-graph", "slot": "canvas"},
                figure=fig,
                style={"height": "100%", "width": "100%", "minWidth": "0"},
                config={"displaylogo": False, "responsive": True},
                responsive=True,
            )
            cap_status = _curve_cap_status_from_figure(fig)
            if not cap_status:
                return graph_canvas
            return html.Div(
                [
                    html.Div(graph_canvas, style={"flex": "1 1 auto", "minHeight": "0"}),
                    html.Div(cap_status, className="rk-log-name"),
                ],
                style={"display": "flex", "flexDirection": "column", "flex": "1 1 auto", "minHeight": "0", "width": "100%"},
            )

        if viz_type == "histogram":
            cols = infer_columns(rows)
            numeric_cols = infer_numeric_columns(rows)
            fallback_hist = numeric_cols[0] if numeric_cols else (cols[0] if cols else "")
            use_hist = str(req.get("x_col") or req.get("y_col") or fallback_hist)
            fig = render_figure(
                rows,
                presentation=presentation_spec if isinstance(presentation_spec, dict) else None,
                x_col=use_hist,
                y_col=use_hist,
                view_type="histogram",
            )
            if fig is None:
                return "No numeric data for histogram."
            hist_color = str(req.get("line_color_rgb") or req.get("line_color") or "")
            if hist_color:
                for tr in fig.data:
                    if isinstance(tr, go.Histogram):
                        tr.update(marker={"color": hist_color}, name=str(tr.name or "distribution"))
            _apply_2d_style(fig, req, apply_legend=True)
            use_custom_title = _flag_on(req.get("use_plot_title"), default=False)
            plot_title = str(req.get("plot_title") or "").strip()
            x_title = str(req.get("x_title") or "").strip()
            y_title = str(req.get("y_title") or "").strip()
            if use_custom_title and plot_title:
                fig.update_layout(title=plot_title)
            if x_title:
                fig.update_xaxes(title_text=x_title)
            if y_title:
                fig.update_yaxes(title_text=y_title)
            fig.update_layout(autosize=True, height=None)
            graph_canvas = dcc.Graph(
                id={"type": "plot-graph", "slot": "canvas"},
                figure=fig,
                style={"height": "100%", "width": "100%", "minWidth": "0"},
                config={"displaylogo": False, "responsive": True},
                responsive=True,
            )
            return graph_canvas

        fig3d = render_figure(
            rows,
            presentation=presentation_spec if isinstance(presentation_spec, dict) else None,
            x_col=str(req.get("x_col") or ""),
            y_col=str(req.get("y_col") or ""),
            z_col=str(req.get("z_col") or ""),
            color_col=str(req.get("color_col") or ""),
            view_type="scatter3d",
        )
        if fig3d is None:
            return "No 3D data."
        marker_size = _parse_float(req.get("marker_size"), 6.0)
        fixed_color = str(req.get("line_color_rgb") or req.get("line_color") or "")
        color_by = str(req.get("color_col") or "")
        for tr in fig3d.data:
            if isinstance(tr, go.Scatter3d):
                marker_update: dict[str, Any] = {}
                if marker_size is not None:
                    marker_update["size"] = float(marker_size)
                if fixed_color and not color_by:
                    marker_update["color"] = fixed_color
                if marker_update:
                    tr.update(marker=marker_update)
        _apply_3d_style(fig3d, req, apply_legend=True)
        use_custom_title = _flag_on(req.get("use_plot_title"), default=False)
        plot_title = str(req.get("plot_title") or "").strip()
        x_title = str(req.get("x_title") or "").strip()
        y_title = str(req.get("y_title") or "").strip()
        z_title = str(req.get("z_title") or "").strip()
        if use_custom_title and plot_title:
            fig3d.update_layout(title=plot_title)
        if x_title or y_title or z_title:
            fig3d.update_scenes(
                xaxis_title=x_title or None,
                yaxis_title=y_title or None,
                zaxis_title=z_title or None,
            )
        fig3d.update_layout(autosize=True, height=None)
        graph3d_canvas = dcc.Graph(
            id={"type": "plot-graph", "slot": "canvas"},
            figure=fig3d,
            style={"height": "100%", "width": "100%", "minWidth": "0"},
            config={"displaylogo": False, "responsive": True},
            responsive=True,
        )
        return graph3d_canvas

    @app.callback(
        Output({"type": "plot-graph", "slot": "canvas"}, "figure"),
        Input({"type": "plot-graph", "slot": "canvas"}, "relayoutData"),
        State("session-store", "data"),
        State("result-store", "data"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def refine_plot2d_on_relayout(
        relayout_data: dict[str, Any] | None,
        session: dict[str, Any] | None,
        result_store: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
    ):
        x_range, has_x_change = _extract_relayout_x_range(relayout_data)
        if not has_x_change:
            return no_update

        source_node = _resolve_canvas_source_node(snapshot, session, None)
        if not isinstance(source_node, dict):
            return no_update
        if str(source_node.get("kind")) == "utility":
            return no_update

        req = source_node.get("request", {}) if isinstance(source_node.get("request"), dict) else {}
        viz_type = str(req.get("visualization_type") or "plot2d").lower()
        if viz_type != "plot2d":
            return no_update

        source_node_id = str(source_node.get("id") or "")
        artifact = _find_source_artifact(snapshot, source_node_id, result_store or {})
        rows = _artifact_rows(artifact if isinstance(artifact, dict) else None)
        if not rows:
            return no_update
        rows = _apply_row_filters(rows, req.get("row_filters"))
        if not rows:
            return no_update

        presentation_spec = None
        meta = source_node.get("metadata", {})
        if isinstance(meta, dict):
            presentation_spec = meta.get("presentation_spec")

        artifact_id = str((artifact or {}).get("id") if isinstance(artifact, dict) else "")
        max_points = _UI_PLOT2D_ZOOM_MAX_POINTS if x_range is not None else _UI_PLOT2D_INITIAL_MAX_POINTS
        cache_scope = "zoom-range" if x_range is not None else "zoom-reset"
        fig = _build_plot2d_figure(
            rows=rows,
            artifact_id=artifact_id,
            presentation_spec=presentation_spec if isinstance(presentation_spec, dict) else None,
            req=req,
            x_col=None,
            y_col=None,
            group_col=None,
            x_range=x_range,
            max_points_total=max_points,
            cache_scope=cache_scope,
        )
        if fig is None:
            return no_update
        return fig

