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

