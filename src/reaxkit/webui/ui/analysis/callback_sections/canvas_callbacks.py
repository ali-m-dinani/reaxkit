"""Register canvas/output callback section for analysis UI.

This module contains a responsibility-focused subset of analysis callback
registrations extracted from `reaxkit.webui.ui.analysis.callbacks`.

**Usage context**

- Dataset browse/workspace and engine-role UI callbacks.
- Canvas render/export and relayout refinement callback wiring.
"""

from __future__ import annotations

from typing import Any

from reaxkit.webui.backend.api import WebUIApiService
from reaxkit.webui.ui.analysis.callback_helpers import *  # noqa: F401,F403
from reaxkit.webui.ui.shared.workspace import empty_state


def _displayed_figure(figures, identifiers, relayouts):
    """Save the visible summary and current camera/axes, without re-reading rows."""
    for index, (identifier, value) in enumerate(zip(identifiers or [], figures or [])):
        if not isinstance(identifier, dict) or identifier.get('slot') != 'canvas' or not isinstance(value, dict) or not value.get('data'):
            continue
        figure = go.Figure(value)
        event = ((relayouts or [])[index] if index < len(relayouts or []) else {}) or {}
        for axis in ('xaxis', 'yaxis'):
            limits = event.get(axis + '.range')
            if axis + '.range[0]' in event and axis + '.range[1]' in event:
                limits = [event[axis + '.range[0]'], event[axis + '.range[1]']]
            if event.get(axis + '.autorange'):
                figure.layout[axis].autorange = True
            elif limits is not None:
                figure.layout[axis].update(range=limits, autorange=False)
        if isinstance(event.get('scene.camera'), dict):
            figure.update_scenes(camera=event['scene.camera'])
        # Theme toggles are client-local relayouts. Preserve their appearance in exports.
        for key, value in event.items():
            if key.endswith(('color', 'bgcolor')) and not key.startswith('shapes'):
                try:
                    figure.layout[key] = value
                except (ValueError, KeyError):
                    pass
        return figure
    return None


def register_canvas_callbacks(app, service: WebUIApiService) -> None:
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
    register_canvas_callbacks(app, service)
    ```
    Sample output:
    `None`
    Meaning:
    Callback handlers for this section are attached to the Dash app.
    """
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
        State({"type": "plot-graph", "slot": ALL}, "relayoutData"),
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
        graph_relayouts: list[dict[str, Any] | None] | None,
    ):
        trig = str(ctx.triggered_id or "")
        if trig not in {"btn-canvas-primary", "btn-canvas-secondary"}:
            return no_update, no_update, no_update
        if trig == "btn-canvas-primary" and int(n_primary or 0) <= 0:
            return no_update, no_update, no_update
        if trig == "btn-canvas-secondary" and int(n_secondary or 0) <= 0:
            return no_update, no_update, no_update

        source = _resolve_canvas_source_node(snapshot, session, None)
        request = (source or {}).get('request') or {}
        if source and (source.get('kind') == 'utility' or request.get('visualization_type') == 'table'):
            artifact = _find_source_artifact(snapshot, source['id'], result_store or {})
            if not artifact:
                return 'No table to export.', no_update, no_update
            export_dir = _default_export_dir(snapshot, session, config)
            name = f"table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            path = export_dir / name
            if trig == 'btn-canvas-secondary':
                path = _browse_save_file(title='Export full table as...', initial_dir=export_dir,
                    initial_name=name, filetypes=[('CSV', '*.csv'), ('Excel', '*.xlsx')], default_ext='.csv')
                if path is None:
                    return 'Export canceled.', no_update, no_update
            job = service.jobs.submit(session['pipeline_id'], 'export_table',
                {'artifact_id': artifact['id'], 'path': str(path), 'row_filters': request.get('row_filters', [])})
            message = f"Export queued: {job['id']}"
            return message, message, no_update

        fig = _displayed_figure(graph_figures, graph_ids, graph_relayouts)
        spinner_tick = html.Span(str(int(n_primary or 0) + int(n_secondary or 0)), style={"display": "none"})
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

        export_dir = _default_export_dir(snapshot, session, config)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
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
        State("plot-context", "data", allow_optional=True),
    )
    def render_result_views(
        session: dict[str, Any] | None,
        result_store: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
        previous_context: dict[str, Any] | None = None,
    ):
        if not session:
            return empty_state("Explore your simulation", "Open a dataset from the hierarchy to get started.")
        node_id = str(session.get("selected_node_id") or "")
        if previous_context and previous_context.get('node_id') != node_id and not node_id.startswith('virtual:visualization:'):
            service.jobs.cancel_view(previous_context['pipeline_id'], previous_context.get('view_id'))
        nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
        selected_node = nodes.get(node_id) if isinstance(nodes, dict) else None
        if isinstance(selected_node, dict) and str(selected_node.get("kind")) == "analysis":
            status = selected_node.get('status')
            if status in ('running', 'queued'):
                return empty_state("Analysis in progress", "Follow progress in Activity, or select an existing presentation while this runs.", kind='running')
            if status == 'error':
                return empty_state("Analysis needs attention", "Inspect error details in Activity, adjust parameters, then run again.", kind='error')
            return empty_state("Choose a presentation", "Select a presentation beneath this analysis, or configure parameters and run it.")
        if str(node_id).startswith("virtual:visualization:"):
            # Selecting the virtual Presentation folder should not change the canvas view.
            return no_update

        source_node = _resolve_canvas_source_node(snapshot, session, None)
        if not isinstance(source_node, dict):
            return empty_state("Explore your simulation", "Choose a presentation in the hierarchy to inspect results.")
        source_node_id = str(source_node.get("id") or "")
        from reaxkit.webui.ui.analysis.callback_sections.table_callbacks import result_table
        artifact = _find_source_artifact(snapshot, source_node_id, result_store or {})
        req = source_node.get('request') or {}
        if artifact and (source_node.get('kind') == 'utility' or req.get('visualization_type') == 'table'):
            return result_table(service, session['pipeline_id'], artifact, req.get('row_filters'))
        if not artifact:
            return empty_state("Waiting for results", "Run the source analysis. Progress and errors appear in Activity.")
        from reaxkit.webui.ui.analysis.callback_sections.plot_callbacks import plot_context, result_plot
        context = plot_context(session['pipeline_id'], artifact, source_node)
        if not context:
            return empty_state("No plottable table", "This result has no tabular data. Choose another result or presentation.")
        if previous_context and previous_context.get('key') == context['key']:
            return no_update
        if previous_context:
            service.jobs.cancel_view(previous_context['pipeline_id'], previous_context.get('view_id'))
        return result_plot(context)
