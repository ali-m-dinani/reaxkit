"""Register pipeline/session callback section for analysis UI.

This module contains a responsibility-focused subset of analysis callback
registrations extracted from `reaxkit.webui.ui.analysis.callbacks`.

**Usage context**

- Snapshot and dataset lifecycle callback wiring.
- Node tree/session selection and status callback wiring.
"""

from __future__ import annotations

from typing import Any

from reaxkit.webui.backend.api import WebUIApiService
from reaxkit.webui.ui.analysis.callback_helpers import *  # noqa: F401,F403


def register_pipeline_callbacks(app, service: WebUIApiService) -> None:
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
    register_pipeline_callbacks(app, service)
    ```
    Sample output:
    `None`
    Meaning:
    Callback handlers for this section are attached to the Dash app.
    """
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

