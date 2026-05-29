"""Register visualization-parameter callback section for analysis UI.

This module contains a responsibility-focused subset of analysis callback
registrations extracted from `reaxkit.webui.ui.analysis.callbacks`.

**Usage context**

- Visualization request parameter save/live-sync callbacks.
- Curve styling, row-filter, and visualization-type callback wiring.
"""

from __future__ import annotations

from typing import Any

from reaxkit.webui.backend.api import WebUIApiService
from reaxkit.webui.ui.analysis.callback_helpers import *  # noqa: F401,F403


def register_visualization_callbacks(app, service: WebUIApiService) -> None:
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
    register_visualization_callbacks(app, service)
    ```
    Sample output:
    `None`
    Meaning:
    Callback handlers for this section are attached to the Dash app.
    """
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

