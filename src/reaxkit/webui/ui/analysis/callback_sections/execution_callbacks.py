"""Register execution and utility callback section for analysis UI.

This module contains a responsibility-focused subset of analysis callback
registrations extracted from `reaxkit.webui.ui.analysis.callbacks`.

**Usage context**

- Request save/live-edit wiring for utility and analysis nodes.
- Node execution/apply callback wiring and result-store updates.
"""

from __future__ import annotations

from typing import Any

from reaxkit.webui.backend.api import WebUIApiService
from reaxkit.webui.ui.analysis.callback_helpers import *  # noqa: F401,F403


def register_execution_callbacks(app, service: WebUIApiService) -> None:
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
    register_execution_callbacks(app, service)
    ```
    Sample output:
    `None`
    Meaning:
    Callback handlers for this section are attached to the Dash app.
    """
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
