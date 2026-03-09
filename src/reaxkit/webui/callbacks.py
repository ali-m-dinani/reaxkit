"""Dash callbacks for Phase 4 workflow (pipeline, utilities, and 3D views)."""

from __future__ import annotations

from typing import Any

from dash import ALL, Input, Output, State, ctx, dash_table, dcc, html, no_update
import plotly.graph_objects as go
import os

from reaxkit.core.analysis_cli_routing_registry import get_registered_analysis_commands
from reaxkit.presentation.specs import ensure_presentation_spec, spec_to_dash_request
from reaxkit.webui.presentation.registry import render_figure
from reaxkit.webui.backend.api import WebUIApiService


def _selected_node(snapshot: dict[str, Any] | None, session: dict[str, Any] | None) -> dict[str, Any] | None:
    if not snapshot or not session:
        return None
    node_id = session.get("selected_node_id")
    nodes = snapshot.get("nodes", {})
    if not isinstance(nodes, dict):
        return None
    node = nodes.get(node_id)
    return node if isinstance(node, dict) else None


def _parse_csv_ints(raw: str | None) -> list[int] | None:
    if raw is None or str(raw).strip() == "":
        return None
    out: list[int] = []
    for part in str(raw).split(","):
        tok = part.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            continue
    return out or None


def _parse_csv_strs(raw: str | None) -> list[str] | None:
    if raw is None or str(raw).strip() == "":
        return None
    out = [p.strip() for p in str(raw).split(",") if p.strip()]
    return out or None


def _artifact_rows(artifact: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not artifact or not isinstance(artifact, dict):
        return []
    payload = artifact.get("payload", {})
    if not isinstance(payload, dict):
        return []
    for value in payload.values():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value
    return []


def _result_cache_from_snapshot(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    if not snapshot or not isinstance(snapshot, dict):
        return {}
    artifacts = snapshot.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return {}
    out: dict[str, Any] = {}
    for artifact in artifacts.values():
        if not isinstance(artifact, dict):
            continue
        node_id = artifact.get("node_id")
        if node_id:
            out[str(node_id)] = artifact
    return out


def _latest_node_id(snapshot: dict[str, Any], kinds: tuple[str, ...]) -> str | None:
    nodes = snapshot.get("nodes", {})
    if not isinstance(nodes, dict):
        return None
    candidates = [n for n in nodes.values() if isinstance(n, dict) and str(n.get("kind")) in kinds]
    if not candidates:
        return None
    candidates.sort(key=lambda n: str(n.get("updated_at", "")))
    return str(candidates[-1].get("id"))


def _ancestor_analysis_id(nodes: dict[str, Any], node_id: str) -> str | None:
    current = nodes.get(node_id)
    seen: set[str] = set()
    while isinstance(current, dict):
        cid = str(current.get("id"))
        if cid in seen:
            return None
        seen.add(cid)
        if str(current.get("kind")) == "analysis":
            return cid
        parent_id = current.get("parent_id")
        if not parent_id:
            return None
        current = nodes.get(str(parent_id))
    return None


def _find_source_artifact(
    snapshot: dict[str, Any] | None,
    selected_node_id: str | None,
    result_store: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not snapshot or not selected_node_id:
        return None
    nodes = snapshot.get("nodes", {})
    artifacts = snapshot.get("artifacts", {})
    if not isinstance(nodes, dict):
        return None
    cache = result_store or {}

    # 1) direct cache
    direct = cache.get(selected_node_id)
    if isinstance(direct, dict):
        return direct

    # 2) walk to nearest ancestor with artifact reference
    current = nodes.get(str(selected_node_id))
    seen: set[str] = set()
    while isinstance(current, dict):
        cid = str(current.get("id"))
        if cid in seen:
            break
        seen.add(cid)
        result_ref = current.get("result_ref")
        if result_ref:
            art = cache.get(cid)
            if isinstance(art, dict):
                return art
            if isinstance(artifacts, dict):
                snap_art = artifacts.get(str(result_ref))
                if isinstance(snap_art, dict):
                    return snap_art
        meta = current.get("metadata", {})
        if isinstance(meta, dict):
            last_id = meta.get("last_artifact_id")
            if last_id and isinstance(artifacts, dict):
                snap_art = artifacts.get(str(last_id))
                if isinstance(snap_art, dict):
                    return snap_art
        parent_id = current.get("parent_id")
        if not parent_id:
            break
        current = nodes.get(str(parent_id))
    return None


def _browse_directory() -> str | None:
    """Open a native folder picker dialog and return selected path."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title="Select dataset folder")
        root.destroy()
        if path and isinstance(path, str):
            return path
    except Exception:
        return None
    return None


def _analysis_dropdown_options(search_value: str | None = None) -> tuple[list[dict[str, Any]], str]:
    """Build grouped analysis dropdown options with search support."""
    # Phase scope: keep only currently-enabled analyses.
    enabled_tasks = {"msd"}
    query = str(search_value or "").strip().lower()
    grouped: dict[str, list[str]] = {}
    for task_name, spec in get_registered_analysis_commands().items():
        if task_name not in enabled_tasks:
            continue
        module_leaf = str(spec.module_path).split(".")[-1]
        group = module_leaf.replace("_workflow", "").strip() or "analysis"
        if query and query not in str(task_name).lower() and query not in group.lower():
            continue
        grouped.setdefault(group, []).append(str(task_name))

    options: list[dict[str, Any]] = []
    first_value = ""
    for group_name in sorted(grouped.keys()):
        options.append(
            {
                "label": html.Span(group_name, style={"fontWeight": "700"}),
                "value": f"__group__:{group_name}",
                "disabled": True,
            }
        )
        for task_name in sorted(grouped[group_name]):
            options.append(
                {
                    "label": html.Span(task_name, style={"paddingLeft": "18px", "display": "inline-block"}),
                    "value": task_name,
                }
            )
            if not first_value:
                first_value = task_name

    if not options or not first_value:
        options = [{"label": "msd", "value": "msd"}]
        first_value = "msd"
    return options, first_value


def _visualization_nodes_for_analysis(snapshot: dict[str, Any] | None, analysis_id: str) -> list[dict[str, Any]]:
    nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
    if not isinstance(nodes, dict):
        return []
    out = [
        n
        for n in nodes.values()
        if isinstance(n, dict)
        and str(n.get("kind")) == "visualization"
        and _ancestor_analysis_id(nodes, str(n.get("id"))) == analysis_id
    ]
    out.sort(key=lambda n: str(n.get("created_at", "")))
    return out


def _build_plot(
    rows: list[dict[str, Any]],
    *,
    x_col: str | None = None,
    y_col: str | None = None,
    group_col: str | None = None,
    line_color: str | None = None,
    line_width: float | None = None,
) -> go.Figure:
    fig = go.Figure()
    if not rows:
        fig.update_layout(title="No plottable data")
        return fig

    groups: dict[str, list[tuple[float, float]]] = {}
    use_x = x_col or ("iter" if "iter" in rows[0] else "frame_index")
    use_y = y_col or ("msd" if "msd" in rows[0] else None)
    if use_y is None:
        fig.update_layout(title="No Y column selected")
        return fig
    for row in rows:
        x_val = row.get(use_x)
        y_val = row.get(use_y)
        if x_val is None or y_val is None:
            continue
        try:
            x = float(x_val)
            y = float(y_val)
        except Exception:
            continue
        key = str(row.get(group_col, "all")) if group_col else "all"
        groups.setdefault(key, []).append((x, y))

    if not groups:
        fig.update_layout(title="No plottable numeric columns")
        return fig

    for atom_id, points in groups.items():
        points.sort(key=lambda t: t[0])
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                mode="lines",
                name=f"atom {atom_id}",
                line={
                    "color": str(line_color) if line_color else None,
                    "width": float(line_width) if line_width is not None else 2.0,
                },
            )
        )
    fig.update_layout(
        title=f"{use_y} vs {use_x}",
        xaxis_title=use_x,
        yaxis_title=use_y,
        template="plotly_white",
    )
    return fig


def _build_3d(
    rows: list[dict[str, Any]],
    *,
    x_col: str | None = None,
    y_col: str | None = None,
    z_col: str | None = None,
    color_col: str | None = None,
) -> go.Figure:
    fig = go.Figure()
    if not rows:
        fig.update_layout(title="No 3D data")
        return fig

    cols = list(rows[0].keys())
    use_x = x_col or ("x" if "x" in cols else ("iter" if "iter" in cols else "frame_index"))
    use_y = y_col or ("y" if "y" in cols else ("atom_id" if "atom_id" in cols else use_x))
    use_z = z_col or ("z" if "z" in cols else ("msd" if "msd" in cols else use_y))

    xvals: list[float] = []
    yvals: list[float] = []
    zvals: list[float] = []
    colors: list[float] = []
    text: list[str] = []
    for r in rows:
        try:
            xv = float(r.get(use_x, 0.0))
            yv = float(r.get(use_y, 0.0))
            zv = float(r.get(use_z, 0.0))
        except Exception:
            continue
        xvals.append(xv)
        yvals.append(yv)
        zvals.append(zv)
        if color_col:
            try:
                colors.append(float(r.get(color_col, 0.0)))
            except Exception:
                colors.append(0.0)
        text.append(str(r.get("atom_id", "")))

    marker: dict[str, Any] = {"size": 4, "opacity": 0.8}
    if color_col:
        marker.update({"color": colors, "colorscale": "Viridis", "colorbar": {"title": color_col}})
    fig.add_trace(
        go.Scatter3d(
            x=xvals,
            y=yvals,
            z=zvals,
            mode="markers",
            marker=marker,
            text=text,
            name="points",
        )
    )
    fig.update_layout(
        template="plotly_white",
        scene={"xaxis_title": use_x, "yaxis_title": use_y, "zaxis_title": use_z},
        title=f"3D View: {use_x}, {use_y}, {use_z}",
    )
    return fig


def _render_pipeline_tree(snapshot: dict[str, Any], selected_node_id: str | None) -> list[Any]:
    nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
    if not isinstance(nodes, dict):
        nodes = {}
    dataset_node = next((n for n in nodes.values() if isinstance(n, dict) and n.get("kind") == "dataset"), None)
    engine_text = "(not loaded)"
    if dataset_node:
        meta = dataset_node.get("metadata", {})
        dataset = meta.get("dataset", {}) if isinstance(meta, dict) else {}
        engine_text = str(dataset.get("engine_override") or dataset.get("engine_detected") or "(not loaded)")

    analysis_nodes = [n for n in nodes.values() if isinstance(n, dict) and n.get("kind") == "analysis"]
    utility_nodes = [n for n in nodes.values() if isinstance(n, dict) and n.get("kind") == "utility"]
    viz_nodes = [n for n in nodes.values() if isinstance(n, dict) and n.get("kind") == "visualization"]

    utilities_by_analysis: dict[str, list[dict[str, Any]]] = {}
    for node in utility_nodes:
        aid = _ancestor_analysis_id(nodes, str(node.get("id")))
        if aid:
            utilities_by_analysis.setdefault(aid, []).append(node)
    visualizations_by_analysis: dict[str, list[dict[str, Any]]] = {}
    for node in viz_nodes:
        aid = _ancestor_analysis_id(nodes, str(node.get("id")))
        if aid:
            visualizations_by_analysis.setdefault(aid, []).append(node)

    def row(node_id: str, label: str, depth: int, status: str | None = None) -> Any:
        selected = str(node_id) == str(selected_node_id)
        prefix = "    " * depth + ("└─ " if depth > 0 else "")
        cls = "rk-tree-node selected" if selected else "rk-tree-node"
        return html.Button(
            [
                html.Span(prefix, className="rk-tree-prefix"),
                html.Span("📁", className="rk-tree-icon"),
                html.Span(label, className="rk-tree-label"),
                html.Span(f"[{status}]" if status else "", className="rk-tree-status"),
            ],
            id={"type": "pipeline-node-btn", "node_id": node_id},
            n_clicks=0,
            className=cls,
        )

    rendered: list[Any] = [
        row("virtual:dataset", "Dataset", 0),
        row("virtual:engine", f"Engine: {engine_text}", 1),
        row("virtual:analysis", "Analysis", 2),
    ]
    for node in analysis_nodes:
        aid = str(node.get("id"))
        rendered.append(row(aid, str(node.get("name", "analysis")), 3, str(node.get("status", "idle"))))
        rendered.append(row(f"virtual:utilities:{aid}", "Utilities", 4))
        for unode in utilities_by_analysis.get(aid, []):
            rendered.append(row(str(unode.get("id")), str(unode.get("name", "utility")), 5, str(unode.get("status", "idle"))))
        rendered.append(row(f"virtual:visualization:{aid}", "Visualization", 4))
        typed_counts: dict[str, int] = {}
        for vnode in visualizations_by_analysis.get(aid, []):
            req = vnode.get("request", {}) if isinstance(vnode.get("request"), dict) else {}
            raw_type = str(req.get("visualization_type") or vnode.get("name") or "viz").lower()
            label_type = {
                "plot2d": "plot 2d",
                "scatter3d": "scatter 3d",
                "histogram": "histogram",
                "table": "table",
            }.get(raw_type, raw_type)
            typed_counts[label_type] = typed_counts.get(label_type, 0) + 1
            tag = f"{typed_counts[label_type]:02d}"
            rendered.append(row(str(vnode.get("id")), f"{label_type}: {tag}", 5, str(vnode.get("status", "idle"))))
    return rendered


def register_callbacks(app, service: WebUIApiService) -> None:
    """Register all Dash callbacks for Phase 4."""

    @app.callback(
        Output("session-store", "data"),
        Output("pipeline-store", "data"),
        Output("result-store", "data"),
        Output("config-store", "data"),
        Output("status-banner", "children"),
        Input("app-init", "n_intervals"),
        prevent_initial_call=False,
    )
    def on_app_init(_: int):
        pipeline = service.create_pipeline({"name": "ReaxKit Pipeline"})
        snapshot = service.get_pipeline(str(pipeline["id"]))
        return (
            {"pipeline_id": pipeline["id"], "selected_node_id": "virtual:dataset"},
            snapshot,
            {},
            {"dataset_path": os.getcwd(), "engine_name": "autodetect", "role_xmolout": "xmolout"},
            "Ready",
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
        cfg["dataset_path"] = str(value or ".")
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
            return no_update, no_update, no_update

        pipeline_id = str(session["pipeline_id"])
        cfg = dict(config or {})
        dataset_path = str(cfg.get("dataset_path") or ".")
        engine_name = str(cfg.get("engine_name") or "autodetect")
        role_xmolout = str(cfg.get("role_xmolout") or "xmolout")
        run_dir = str(dataset_path or ".").strip() or "."
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
            },
        )
        snapshot = service.get_pipeline(pipeline_id)
        return {"pipeline_id": pipeline_id, "selected_node_id": dataset_node["id"]}, snapshot, "Dataset loaded"

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
    def on_add_msd_node(
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

        nodes = snapshot.get("nodes", {})
        dataset_node = None
        if isinstance(nodes, dict):
            dataset_node = next((n for n in nodes.values() if isinstance(n, dict) and n.get("kind") == "dataset"), None)
        if not dataset_node:
            return no_update, no_update, "WARN: Load a dataset first"

        task = str(analysis_type or "msd").strip().lower()
        if task != "msd":
            return no_update, no_update, "WARN: Only msd is enabled right now"

        msd_node = service.add_node(
            pipeline_id,
            {
                "parent_id": dataset_node["id"],
                "kind": "analysis",
                "name": task,
                "metadata": {"task_name": task},
                "request": {
                    "atom_ids": None,
                    "atom_types": None,
                    "dims": ["x", "y", "z"],
                    "origin": "first",
                    "frames": None,
                    "every": 1,
                },
            },
        )
        next_snapshot = service.get_pipeline(pipeline_id)
        next_session = {"pipeline_id": pipeline_id, "selected_node_id": msd_node["id"]}
        return next_session, next_snapshot, "MSD node added"

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Output("pipeline-store", "data", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("btn-add-filter-node", "n_clicks"),
        Input("btn-add-ema-node", "n_clicks"),
        Input("btn-add-sma-node", "n_clicks"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def on_add_utility_node(
        n_filter: int,
        n_ema: int,
        n_sma: int,
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
    ):
        if not session or not snapshot:
            return no_update, no_update, no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        selected_node_id = str(session.get("selected_node_id") or "")
        node = _selected_node(snapshot, session)
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
                nodes = snapshot.get("nodes", {})
                node = nodes.get(parent_id) if isinstance(nodes, dict) else None
            if not node:
                return no_update, no_update, "WARN: Select a parent node first"

        trig = ctx.triggered_id
        util_name = None
        request = {}
        label = ""
        if trig == "btn-add-filter-node":
            util_name = "filter_rows"
            request = {"column": "atom_id", "values": ""}
            label = "Filter utility"
        elif trig == "btn-add-ema-node":
            util_name = "denoise_ema"
            request = {"column": "msd", "alpha": 0.3, "group_by": "atom_id", "x_col": "iter"}
            label = "EMA utility"
        elif trig == "btn-add-sma-node":
            util_name = "denoise_sma"
            request = {"column": "msd", "window": 5, "group_by": "atom_id", "x_col": "iter"}
            label = "SMA utility"
        if util_name is None:
            return no_update, no_update, no_update

        util_node = service.add_node(
            pipeline_id,
            {
                "parent_id": node["id"],
                "kind": "utility",
                "name": util_name,
                "metadata": {"utility_name": util_name},
                "request": request,
            },
        )
        next_snapshot = service.get_pipeline(pipeline_id)
        next_session = {"pipeline_id": pipeline_id, "selected_node_id": util_node["id"]}
        return next_session, next_snapshot, f"{label} added"

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
                    "color_col": str(color_col or ""),
                    "group_col": str(group_col or ""),
                    "line_color": "blue",
                    "line_color_rgb": "",
                    "line_width": 2.0,
                    "table_filter_col": "",
                    "table_filter_value": "",
                    "table_max_rows": 200,
                },
            },
        )
        next_snapshot = service.get_pipeline(pipeline_id)
        return {"pipeline_id": pipeline_id, "selected_node_id": node["id"]}, next_snapshot, "Visualization added"

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
        Input("btn-save-node-params", "n_clicks"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        State("msd-atom-ids", "value"),
        State("msd-atom-types", "value"),
        State("msd-dims", "value"),
        State("msd-origin", "value"),
        State("msd-frames", "value"),
        State("msd-every", "value"),
        prevent_initial_call=True,
    )
    def on_save_node_params(
        n_clicks: int,
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
        atom_ids_raw: str | None,
        atom_types_raw: str | None,
        dims_raw: str | None,
        origin_raw: str | None,
        frames_raw: str | None,
        every_raw: str | None,
    ):
        if not n_clicks or not session or not snapshot:
            return no_update, no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        node = _selected_node(snapshot, session)
        if not pipeline_id or not node or node.get("kind") != "analysis":
            return no_update, "WARN: Select an analysis node"

        dims = _parse_csv_strs(dims_raw) or ["x", "y", "z"]
        origin: str | int = str(origin_raw or "first").strip() or "first"
        if origin != "first":
            try:
                origin = int(origin)
            except ValueError:
                origin = "first"
        try:
            every = max(1, int(str(every_raw or "1").strip()))
        except ValueError:
            every = 1

        request_payload = {
            "atom_ids": _parse_csv_ints(atom_ids_raw),
            "atom_types": _parse_csv_strs(atom_types_raw),
            "dims": dims,
            "origin": origin,
            "frames": _parse_csv_ints(frames_raw),
            "every": every,
        }
        service.update_node(pipeline_id, str(node["id"]), {"request": request_payload})
        next_snapshot = service.get_pipeline(pipeline_id)
        return next_snapshot, "Parameters saved"

    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("btn-save-util-params", "n_clicks"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        State("util-filter-column", "value"),
        State("util-filter-values", "value"),
        State("util-denoise-column", "value"),
        State("util-denoise-alpha", "value"),
        State("util-denoise-window", "value"),
        State("util-denoise-group", "value"),
        State("util-denoise-xcol", "value"),
        prevent_initial_call=True,
    )
    def on_save_utility_params(
        n_clicks: int,
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
        filter_column: str | None,
        filter_values: str | None,
        denoise_column: str | None,
        denoise_alpha: float | None,
        denoise_window: int | None,
        denoise_group: str | None,
        denoise_xcol: str | None,
    ):
        if not n_clicks or not session or not snapshot:
            return no_update, no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        node = _selected_node(snapshot, session)
        if not pipeline_id or not node or node.get("kind") != "utility":
            return no_update, "WARN: Select a utility node"

        util_name = str(node.get("metadata", {}).get("utility_name") or node.get("name") or "").lower()
        if util_name in {"filter_rows", "filter_atoms"}:
            request_payload = {"column": filter_column or "atom_id", "values": filter_values or ""}
        elif util_name == "denoise_ema":
            request_payload = {
                "column": denoise_column or "msd",
                "alpha": float(denoise_alpha if denoise_alpha is not None else 0.3),
                "group_by": denoise_group or "atom_id",
                "x_col": denoise_xcol or "iter",
            }
        elif util_name == "denoise_sma":
            request_payload = {
                "column": denoise_column or "msd",
                "window": int(denoise_window if denoise_window is not None else 5),
                "group_by": denoise_group or "atom_id",
                "x_col": denoise_xcol or "iter",
            }
        else:
            return no_update, "WARN: Unsupported utility node"

        service.update_node(pipeline_id, str(node["id"]), {"request": request_payload})
        next_snapshot = service.get_pipeline(pipeline_id)
        return next_snapshot, "Utility parameters saved"

    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Output("result-store", "data", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("btn-apply-node", "n_clicks"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        State("result-store", "data"),
        prevent_initial_call=True,
    )
    def on_apply_node(
        n_clicks: int,
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
        result_store: dict[str, Any] | None,
    ):
        if not n_clicks or not session or not snapshot:
            return no_update, no_update, no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        node = _selected_node(snapshot, session)
        if not pipeline_id or not node:
            return no_update, no_update, "WARN: Select a node first"

        try:
            run_result = service.apply_node(pipeline_id, str(node["id"]))
        except Exception as exc:
            return no_update, no_update, f"ERROR: Execute failed: {exc}"

        artifact = run_result.get("artifact") if isinstance(run_result, dict) else None
        next_store = dict(result_store or {})
        if isinstance(artifact, dict) and "id" in artifact:
            next_store[str(node["id"])] = artifact
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
        return next_snapshot, next_store, "Node executed"

    @app.callback(
        Output("result-tabs", "children"),
        Output("result-tabs", "value"),
        Input("session-store", "data"),
        Input("pipeline-store", "data"),
        State("result-tabs", "value"),
    )
    def sync_result_tabs(
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
        current_value: str | None,
    ):
        if not session or not snapshot:
            return [], None
        selected_id = str(session.get("selected_node_id") or "")
        nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
        if not isinstance(nodes, dict):
            return [], None

        viz_nodes: list[dict[str, Any]] = []
        if selected_id.startswith("virtual:visualization:"):
            analysis_id = selected_id.split(":", 2)[2]
            viz_nodes = _visualization_nodes_for_analysis(snapshot, analysis_id)
        else:
            node = nodes.get(selected_id)
            if isinstance(node, dict) and str(node.get("kind")) == "visualization":
                viz_nodes = [node]
            elif isinstance(node, dict) and str(node.get("kind")) == "analysis":
                viz_nodes = []

        if not viz_nodes:
            return [], None
        tabs = [dcc.Tab(label=str(v.get("name", "visualization")), value=str(v.get("id"))) for v in viz_nodes]
        valid = {str(v.get("id")) for v in viz_nodes}
        if selected_id in valid:
            return tabs, selected_id
        if current_value and str(current_value) in valid:
            return tabs, str(current_value)
        return tabs, str(viz_nodes[0].get("id"))

    @app.callback(
        Output("table-controls", "style"),
        Output("plot-controls", "style"),
        Output("hist-controls", "style"),
        Output("view3d-controls", "style"),
        Output("sync-controls", "style"),
        Input("result-tabs", "value"),
        Input("pipeline-store", "data"),
    )
    def toggle_result_controls(tab_value: str | None, snapshot: dict[str, Any] | None):
        nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
        node = nodes.get(tab_value) if isinstance(nodes, dict) and tab_value else None
        if not isinstance(node, dict):
            return {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "none"}
        req = node.get("request", {}) if isinstance(node, dict) and isinstance(node.get("request"), dict) else {}
        viz_type = str(req.get("visualization_type") or "plot2d").lower()
        table_style = {"display": "none"}
        plot_style = {"display": "none"}
        hist_style = {"display": "none"}
        view3d_style = {"display": "none"}
        sync_style = {"display": "none"}
        return table_style, plot_style, hist_style, view3d_style, sync_style

    @app.callback(
        Output("table-filter-col", "options"),
        Output("table-filter-col", "value"),
        Output("plot-x-col", "options"),
        Output("plot-x-col", "value"),
        Output("plot-y-col", "options"),
        Output("plot-y-col", "value"),
        Output("plot-group-col", "options"),
        Output("plot-group-col", "value"),
        Output("hist-col", "options"),
        Output("hist-col", "value"),
        Output("view3d-x-col", "options"),
        Output("view3d-x-col", "value"),
        Output("view3d-y-col", "options"),
        Output("view3d-y-col", "value"),
        Output("view3d-z-col", "options"),
        Output("view3d-z-col", "value"),
        Output("view3d-color-col", "options"),
        Output("view3d-color-col", "value"),
        Output("focus-atom", "options"),
        Output("focus-atom", "value"),
        Input("session-store", "data"),
        Input("result-tabs", "value"),
        Input("pipeline-store", "data"),
        Input("result-store", "data"),
    )
    def populate_plot_controls(
        session: dict[str, Any] | None,
        tab_node_id: str | None,
        snapshot: dict[str, Any] | None,
        result_store: dict[str, Any] | None,
    ):
        if not session:
            return [], None, [], None, [], None, [], None, [], None, [], None, [], None, [], None, [], None, [], None
        node_id = str(tab_node_id or session.get("selected_node_id") or "")
        artifact = _find_source_artifact(snapshot, node_id, result_store or {})
        rows = _artifact_rows(artifact if isinstance(artifact, dict) else None)
        if not rows:
            return [], None, [], None, [], None, [], None, [], None, [], None, [], None, [], None, [], None, [], None
        cols = list(rows[0].keys())
        numeric_cols = []
        for col in cols:
            v = rows[0].get(col)
            if isinstance(v, (int, float)):
                numeric_cols.append(col)
        x_default = "iter" if "iter" in cols else ("frame_index" if "frame_index" in cols else (numeric_cols[0] if numeric_cols else None))
        y_default = "msd" if "msd" in cols else (numeric_cols[1] if len(numeric_cols) > 1 else (numeric_cols[0] if numeric_cols else None))
        group_default = "atom_id" if "atom_id" in cols else None
        options_all = [{"label": c, "value": c} for c in cols]
        options_numeric = [{"label": c, "value": c} for c in numeric_cols]
        hist_default = "msd" if "msd" in numeric_cols else (numeric_cols[0] if numeric_cols else None)
        x3d_default = "x" if "x" in cols else x_default
        y3d_default = "y" if "y" in cols else ("atom_id" if "atom_id" in cols else y_default)
        z3d_default = "z" if "z" in cols else ("msd" if "msd" in cols else y_default)
        color3d_default = "msd" if "msd" in numeric_cols else None
        atom_ids = sorted({str(r.get("atom_id")) for r in rows if r.get("atom_id") is not None})
        focus_options = [{"label": aid, "value": aid} for aid in atom_ids]
        return (
            options_all,
            cols[0] if cols else None,
            options_all,
            x_default,
            options_numeric,
            y_default,
            options_all,
            group_default,
            options_numeric,
            hist_default,
            options_all,
            x3d_default,
            options_all,
            y3d_default,
            options_all,
            z3d_default,
            options_numeric,
            color3d_default,
            focus_options,
            None,
        )

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
            return "Parameters: Visualization"
        node = _selected_node(snapshot, session)
        if node:
            return f"Parameters: {str(node.get('name', 'Node'))}"
        return "Parameters"

    @app.callback(
        Output("properties-content", "children"),
        Input("pipeline-store", "data"),
        Input("session-store", "data"),
        Input("result-store", "data"),
    )
    def render_properties(snapshot: dict[str, Any] | None, session: dict[str, Any] | None, result_store_in: dict[str, Any] | None):
        selected_id = str((session or {}).get("selected_node_id") or "")
        result_store = dict(result_store_in or {})

        if selected_id == "virtual:engine":
            return html.Div(
                [
                    html.Label("Engine name"),
                    dcc.Dropdown(
                        id="input-engine-name",
                        options=[
                            {"label": "autodetect", "value": "autodetect"},
                            {"label": "reaxff", "value": "reaxff"},
                            {"label": "ams", "value": "ams"},
                            {"label": "lammps", "value": "lammps"},
                        ],
                        value="autodetect",
                        clearable=False,
                    ),
                    html.Div(
                        [
                            html.Label("xmolout:"),
                            dcc.Input(id="input-role-xmolout", value="xmolout", type="text"),
                        ],
                        id="engine-file-roles",
                        className="rk-stack",
                    ),
                ],
                className="rk-stack",
            )

        if selected_id == "virtual:analysis":
            analysis_options, analysis_default = _analysis_dropdown_options()
            return html.Div(
                [
                    html.Label("Analysis type"),
                    dcc.Dropdown(
                        id="input-analysis-type",
                        options=analysis_options,
                        value=analysis_default,
                        clearable=False,
                    ),
                    html.Button("Add Analysis Node", id="btn-add-analysis-node", n_clicks=0),
                ],
                className="rk-stack",
            )

        if selected_id.startswith("virtual:utilities"):
            return html.Div(
                [
                    html.Div("Node: Utilities"),
                    html.Button("Add Filter Utility", id="btn-add-filter-node", n_clicks=0),
                    html.Button("Add EMA Utility", id="btn-add-ema-node", n_clicks=0),
                    html.Button("Add SMA Utility", id="btn-add-sma-node", n_clicks=0),
                ],
                className="rk-stack",
            )

        if selected_id.startswith("virtual:visualization"):
            snapshot_nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
            cols: list[str] = ["iter", "frame_index", "msd", "atom_id"]
            if isinstance(snapshot_nodes, dict):
                aid = selected_id.split(":", 2)[2] if ":" in selected_id else None
                if aid:
                    # Best effort: infer from latest artifact in the selected analysis subtree.
                    source_art = _find_source_artifact(snapshot, aid, result_store)
                    rows = _artifact_rows(source_art if isinstance(source_art, dict) else None)
                    if rows:
                        cols = list(rows[0].keys())
            opts = [{"label": c, "value": c} for c in cols]
            return html.Div(
                [
                    html.Label("Visualization type"),
                    dcc.Dropdown(
                        id="viz-type",
                        options=[
                            {"label": "plot2d", "value": "plot2d"},
                            {"label": "histogram", "value": "histogram"},
                            {"label": "scatter3d", "value": "scatter3d"},
                            {"label": "table", "value": "table"},
                        ],
                        value="plot2d",
                        clearable=False,
                    ),
                    html.Label("x axis content"),
                    dcc.Dropdown(id="viz-x-col", options=opts, value="iter" if "iter" in cols else cols[0], clearable=False),
                    html.Label("y axis content"),
                    dcc.Dropdown(id="viz-y-col", options=opts, value="msd" if "msd" in cols else cols[0], clearable=False),
                    dcc.Dropdown(id="viz-z-col", options=opts, value=cols[0], clearable=False, style={"display": "none"}),
                    html.Label("color by"),
                    dcc.Dropdown(id="viz-color-col", options=opts, value=None, clearable=True),
                    html.Label("group content"),
                    dcc.Dropdown(id="viz-group-col", options=opts, value="atom_id" if "atom_id" in cols else None, clearable=True),
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
                    ),
                    dcc.Input(id="viz-line-color-rgb", type="text", value="", style={"display": "none"}),
                    dcc.Input(id="viz-line-width", type="number", value=2, style={"display": "none"}),
                    dcc.Dropdown(id="viz-table-filter-col", options=opts, value=None, clearable=True, style={"display": "none"}),
                    dcc.Input(id="viz-table-filter-value", type="text", value="", style={"display": "none"}),
                    dcc.Input(id="viz-table-max-rows", type="number", value=200, style={"display": "none"}),
                    html.Button("Add Visualization", id="btn-add-visualization-node", n_clicks=0, className="rk-btn-exec"),
                ],
                className="rk-stack",
            )

        if selected_id == "virtual:dataset":
            return html.Div(
                [
                    html.Label("Dataset path"),
                    dcc.Input(id="input-dataset-path", value=os.getcwd(), type="text"),
                    html.Button("Browse...", id="btn-browse-dataset", n_clicks=0),
                    html.Button("Load Dataset", id="btn-load-dataset", n_clicks=0),
                    html.Hr(),
                    html.Label("Snapshot path"),
                    dcc.Input(id="input-snapshot-path", value="./reaxkit.pipeline.json", type="text"),
                    html.Button("Save Snapshot", id="btn-save-snapshot", n_clicks=0),
                    html.Button("Load Snapshot", id="btn-load-snapshot", n_clicks=0),
                    html.Label("Bundle output dir"),
                    dcc.Input(id="input-bundle-dir", value="./reaxkit.bundle", type="text"),
                    html.Button("Export Bundle", id="btn-export-bundle", n_clicks=0),
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

        if node.get("kind") == "analysis" and str(node.get("name", "")).lower() == "msd":
            req = node.get("request", {}) if isinstance(node.get("request", {}), dict) else {}
            lines.extend(
                [
                    html.Div(
                        [
                            html.Span("atom_ids (comma-separated)"),
                            html.Span("?", className="rk-help-dot", title="Examples: 1,2,3 or leave empty for all atoms."),
                        ],
                        className="rk-help-inline",
                    ),
                    dcc.Input(
                        id="msd-atom-ids",
                        type="text",
                        value=",".join(str(v) for v in (req.get("atom_ids") or [])),
                    ),
                    html.Div(
                        [
                            html.Span("atom_types (comma-separated)"),
                            html.Span("?", className="rk-help-dot", title="Examples: H,O,C . Used only when atom_ids is empty."),
                        ],
                        className="rk-help-inline",
                    ),
                    dcc.Input(
                        id="msd-atom-types",
                        type="text",
                        value=",".join(str(v) for v in (req.get("atom_types") or [])),
                    ),
                    html.Div(
                        [
                            html.Span("dims"),
                            html.Span("?", className="rk-help-dot", title="Any subset of x,y,z. Example: x,y,z or x,z"),
                        ],
                        className="rk-help-inline",
                    ),
                    dcc.Input(
                        id="msd-dims",
                        type="text",
                        value=",".join(str(v) for v in (req.get("dims") or ["x", "y", "z"])),
                    ),
                    html.Div(
                        [
                            html.Span("origin"),
                            html.Span("?", className="rk-help-dot", title="Use 'first' or a selected frame index (e.g. 0, 100)."),
                        ],
                        className="rk-help-inline",
                    ),
                    dcc.Input(id="msd-origin", type="text", value=str(req.get("origin", "first"))),
                    html.Div(
                        [
                            html.Span("frames (comma-separated)"),
                            html.Span("?", className="rk-help-dot", title="Examples: 0,10,20. Leave empty to use all frames."),
                        ],
                        className="rk-help-inline",
                    ),
                    dcc.Input(
                        id="msd-frames",
                        type="text",
                        value=",".join(str(v) for v in (req.get("frames") or [])),
                    ),
                    html.Div(
                        [
                            html.Span("every"),
                            html.Span("?", className="rk-help-dot", title="Frame stride. 1 means use every frame, 10 means every 10th frame."),
                        ],
                        className="rk-help-inline",
                    ),
                    dcc.Input(id="msd-every", type="number", value=int(req.get("every", 1)), min=1, step=1),
                    html.Div(
                        [
                            html.Button("Save Params", id="btn-save-node-params", n_clicks=0, className="rk-btn-save"),
                            html.Button("Execute", id="btn-apply-node", n_clicks=0, className="rk-btn-exec"),
                        ],
                        className="rk-inline-actions",
                    ),
                ]
            )
            return html.Div(lines, className="rk-stack")

        if node.get("kind") == "utility":
            req = node.get("request", {}) if isinstance(node.get("request", {}), dict) else {}
            util_name = str(node.get("metadata", {}).get("utility_name") or node.get("name") or "").lower()
            lines.append(html.Div(f"Utility: {util_name}"))
            if util_name in {"filter_rows", "filter_atoms"}:
                lines.extend(
                    [
                        html.Label("column"),
                        dcc.Input(id="util-filter-column", type="text", value=str(req.get("column", "atom_id"))),
                        html.Label("values (comma-separated)"),
                        dcc.Input(id="util-filter-values", type="text", value=str(req.get("values", ""))),
                        dcc.Input(id="util-denoise-column", type="text", value="msd", style={"display": "none"}),
                        dcc.Input(id="util-denoise-alpha", type="number", value=0.3, style={"display": "none"}),
                        dcc.Input(id="util-denoise-window", type="number", value=5, style={"display": "none"}),
                        dcc.Input(id="util-denoise-group", type="text", value="atom_id", style={"display": "none"}),
                        dcc.Input(id="util-denoise-xcol", type="text", value="iter", style={"display": "none"}),
                    ]
                )
            elif util_name == "denoise_ema":
                lines.extend(
                    [
                        html.Label("column"),
                        dcc.Input(id="util-denoise-column", type="text", value=str(req.get("column", "msd"))),
                        html.Label("alpha"),
                        dcc.Input(id="util-denoise-alpha", type="number", value=float(req.get("alpha", 0.3)), min=0.01, max=1.0, step=0.01),
                        html.Label("group_by"),
                        dcc.Input(id="util-denoise-group", type="text", value=str(req.get("group_by", "atom_id"))),
                        html.Label("x_col"),
                        dcc.Input(id="util-denoise-xcol", type="text", value=str(req.get("x_col", "iter"))),
                        dcc.Input(id="util-denoise-window", type="number", value=5, style={"display": "none"}),
                    ]
                )
            elif util_name == "denoise_sma":
                lines.extend(
                    [
                        html.Label("column"),
                        dcc.Input(id="util-denoise-column", type="text", value=str(req.get("column", "msd"))),
                        html.Label("window"),
                        dcc.Input(id="util-denoise-window", type="number", value=int(req.get("window", 5)), min=1, step=1),
                        html.Label("group_by"),
                        dcc.Input(id="util-denoise-group", type="text", value=str(req.get("group_by", "atom_id"))),
                        html.Label("x_col"),
                        dcc.Input(id="util-denoise-xcol", type="text", value=str(req.get("x_col", "iter"))),
                        dcc.Input(id="util-denoise-alpha", type="number", value=0.3, style={"display": "none"}),
                    ]
                )
            else:
                lines.append("No editor for this utility yet.")
            lines.extend(
                [
                    dcc.Input(id="util-filter-column", type="text", value="", style={"display": "none"})
                    if util_name not in {"filter_rows", "filter_atoms"}
                    else html.Div(style={"display": "none"}),
                    dcc.Input(id="util-filter-values", type="text", value="", style={"display": "none"})
                    if util_name not in {"filter_rows", "filter_atoms"}
                    else html.Div(style={"display": "none"}),
                    html.Div(
                        [
                            html.Button("Save Params", id="btn-save-util-params", n_clicks=0, className="rk-btn-save"),
                            html.Button("Execute", id="btn-apply-node", n_clicks=0, className="rk-btn-exec"),
                        ],
                        className="rk-inline-actions",
                    ),
                ]
            )
            return html.Div(lines, className="rk-stack")

        if node.get("kind") == "visualization":
            req = node.get("request", {}) if isinstance(node.get("request", {}), dict) else {}
            source_art = _find_source_artifact(snapshot, str(node.get("id")), {})
            source_rows = _artifact_rows(source_art if isinstance(source_art, dict) else None)
            cols = list(source_rows[0].keys()) if source_rows else ["iter", "frame_index", "msd", "atom_id"]
            opts = [{"label": c, "value": c} for c in cols]
            viz_type = str(req.get("visualization_type") or "plot2d")
            color_options = [
                {"label": "blue", "value": "blue"},
                {"label": "red", "value": "red"},
                {"label": "black", "value": "black"},
                {"label": "green", "value": "green"},
                {"label": "orange", "value": "orange"},
                {"label": "purple", "value": "purple"},
            ]
            body: list[Any] = [
                html.Label("Visualization type"),
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
                        dcc.Dropdown(id="viz-x-col", options=opts, value=str(req.get("x_col") or (cols[0] if cols else "")), clearable=False),
                        html.Label("y axis content"),
                        dcc.Dropdown(id="viz-y-col", options=opts, value=str(req.get("y_col") or (cols[0] if cols else "")), clearable=False),
                        html.Label("group content"),
                        dcc.Dropdown(id="viz-group-col", options=opts, value=str(req.get("group_col") or "") or None, clearable=True),
                        html.Div(
                            [
                                html.Label("Line color"),
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
                        html.Label("Line color (RGB, optional)"),
                        dcc.Input(id="viz-line-color-rgb", type="text", value=str(req.get("line_color_rgb") or ""), placeholder="e.g. rgb(255,0,0)"),
                        html.Label("Line width"),
                        dcc.Input(id="viz-line-width", type="number", value=float(req.get("line_width") or 2), min=1, max=8, step=1),
                        dcc.Dropdown(id="viz-z-col", options=opts, value=str(req.get("z_col") or ""), clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-color-col", options=opts, value=str(req.get("color_col") or "") or None, clearable=True, style={"display": "none"}),
                    ]
                )
            elif viz_type == "scatter3d":
                body.extend(
                    [
                        html.Label("x axis content"),
                        dcc.Dropdown(id="viz-x-col", options=opts, value=str(req.get("x_col") or (cols[0] if cols else "")), clearable=False),
                        html.Label("y axis content"),
                        dcc.Dropdown(id="viz-y-col", options=opts, value=str(req.get("y_col") or (cols[0] if cols else "")), clearable=False),
                        html.Label("z axis content"),
                        dcc.Dropdown(id="viz-z-col", options=opts, value=str(req.get("z_col") or (cols[0] if cols else "")), clearable=False),
                        html.Label("color by"),
                        dcc.Dropdown(id="viz-color-col", options=opts, value=str(req.get("color_col") or "") or None, clearable=True),
                        dcc.Dropdown(id="viz-group-col", options=opts, value=str(req.get("group_col") or "") or None, clearable=True, style={"display": "none"}),
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
                    ]
                )
            elif viz_type == "histogram":
                body.extend(
                    [
                        html.Label("value content"),
                        dcc.Dropdown(id="viz-x-col", options=opts, value=str(req.get("x_col") or "msd"), clearable=False),
                        dcc.Dropdown(id="viz-y-col", options=opts, value=str(req.get("y_col") or ""), clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-z-col", options=opts, value=str(req.get("z_col") or ""), clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-color-col", options=opts, value=str(req.get("color_col") or "") or None, clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-group-col", options=opts, value=str(req.get("group_col") or "") or None, clearable=True, style={"display": "none"}),
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
                    ]
                )
            else:
                body.extend(
                    [
                        html.Label("filter column"),
                        dcc.Dropdown(
                            id="viz-table-filter-col",
                            options=opts,
                            value=str(req.get("table_filter_col") or "") or None,
                            clearable=True,
                        ),
                        html.Label("filter value"),
                        dcc.Input(id="viz-table-filter-value", type="text", value=str(req.get("table_filter_value") or "")),
                        html.Label("Visible rows"),
                        dcc.Input(
                            id="viz-table-max-rows",
                            type="number",
                            value=int(req.get("table_max_rows") or 200),
                            min=10,
                            step=10,
                        ),
                        dcc.Dropdown(id="viz-x-col", options=opts, value=str(req.get("x_col") or ""), clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-y-col", options=opts, value=str(req.get("y_col") or ""), clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-z-col", options=opts, value=str(req.get("z_col") or ""), clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-color-col", options=opts, value=str(req.get("color_col") or "") or None, clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-group-col", options=opts, value=str(req.get("group_col") or "") or None, clearable=True, style={"display": "none"}),
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
                    ]
                )
            if viz_type != "table":
                body.extend(
                    [
                        dcc.Input(id="viz-table-filter-value", type="text", value="", style={"display": "none"}),
                        dcc.Dropdown(id="viz-table-filter-col", options=opts, value=None, clearable=True, style={"display": "none"}),
                        dcc.Input(id="viz-table-max-rows", type="number", value=200, style={"display": "none"}),
                    ]
                )
            lines.extend(
                body
            )
            if viz_type in {"scatter3d", "histogram"}:
                lines.append(
                    html.Div(
                        [
                            html.Button("Save Params", id="btn-save-viz-params", n_clicks=0, className="rk-btn-save"),
                            html.Button("Execute", id="btn-apply-node", n_clicks=0, className="rk-btn-exec"),
                        ],
                        className="rk-inline-actions",
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
        Output("input-analysis-type", "options"),
        Output("input-analysis-type", "value"),
        Input("input-analysis-type", "search_value"),
        State("input-analysis-type", "value"),
        prevent_initial_call=False,
    )
    def update_analysis_dropdown(search_value: str | None, current_value: str | None):
        options, default_value = _analysis_dropdown_options(search_value)
        valid_values = {
            str(o.get("value"))
            for o in options
            if isinstance(o, dict) and not str(o.get("value", "")).startswith("__group__:")
        }
        next_value = str(current_value) if current_value and str(current_value) in valid_values else default_value
        return options, next_value

    @app.callback(
        Output("viz-color-name-wrap", "style"),
        Input("viz-line-color-rgb", "value"),
        State("viz-type", "value"),
        prevent_initial_call=False,
    )
    def toggle_viz_color_name(rgb_value: str | None, viz_type: str | None):
        if str(viz_type or "") != "plot2d":
            return {"display": "none"}
        if rgb_value and str(rgb_value).strip():
            return {"display": "none"}
        return {"display": "grid"}

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
        State("viz-line-color-name", "value"),
        State("viz-line-color-rgb", "value"),
        State("viz-line-width", "value"),
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
        line_color_name: str | None,
        line_color_rgb: str | None,
        line_width: float | None,
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
        payload = {
            "visualization_type": str(viz_type or "plot2d"),
            "x_col": str(x_col or ""),
            "y_col": str(y_col or ""),
            "z_col": str(z_col or ""),
            "color_col": str(color_col or ""),
            "group_col": str(group_col or ""),
            "line_color": str(line_color_name or "blue"),
            "line_color_rgb": str(line_color_rgb or ""),
            "line_width": float(line_width if line_width is not None else 2.0),
            "table_filter_col": str(table_filter_col or ""),
            "table_filter_value": str(table_filter_value or ""),
            "table_max_rows": int(table_max_rows) if table_max_rows is not None else 200,
        }
        service.update_node(pipeline_id, str(node["id"]), {"request": payload, "metadata": {"visualization_type": payload["visualization_type"]}})
        next_snapshot = service.get_pipeline(pipeline_id)
        return next_snapshot, "Visualization parameters saved"

    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Input("viz-type", "value"),
        Input("viz-x-col", "value"),
        Input("viz-y-col", "value"),
        Input("viz-z-col", "value"),
        Input("viz-color-col", "value"),
        Input("viz-group-col", "value"),
        Input("viz-line-color-name", "value"),
        Input("viz-line-color-rgb", "value"),
        Input("viz-line-width", "value"),
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
        line_color_name: str | None,
        line_color_rgb: str | None,
        line_width: float | None,
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
    ):
        if not session or not snapshot:
            return no_update
        if str(viz_type or "") != "plot2d":
            return no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        node = _selected_node(snapshot, session)
        if not pipeline_id or not node or node.get("kind") != "visualization":
            return no_update
        payload = {
            "visualization_type": "plot2d",
            "x_col": str(x_col or ""),
            "y_col": str(y_col or ""),
            "z_col": str(z_col or ""),
            "color_col": str(color_col or ""),
            "group_col": str(group_col or ""),
            "line_color": str(line_color_name or "blue"),
            "line_color_rgb": str(line_color_rgb or ""),
            "line_width": float(line_width if line_width is not None else 2.0),
            "table_filter_col": "",
            "table_filter_value": "",
        }
        service.update_node(
            pipeline_id,
            str(node["id"]),
            {"request": payload, "metadata": {"visualization_type": payload["visualization_type"]}},
        )
        return service.get_pipeline(pipeline_id)

    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Input("viz-type", "value"),
        Input("viz-table-filter-col", "value"),
        Input("viz-table-filter-value", "value"),
        Input("viz-table-max-rows", "value"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def on_live_table_params(
        viz_type: str | None,
        table_filter_col: str | None,
        table_filter_value: str | None,
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
        payload["table_filter_col"] = str(table_filter_col or "")
        payload["table_filter_value"] = str(table_filter_value or "")
        payload["table_max_rows"] = int(table_max_rows) if table_max_rows is not None else int(req_old.get("table_max_rows") or 200)
        service.update_node(
            pipeline_id,
            str(node["id"]),
            {"request": payload, "metadata": {"visualization_type": payload["visualization_type"]}},
        )
        return service.get_pipeline(pipeline_id)

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
        nodes = snapshot.get("nodes", {})
        if not isinstance(nodes, dict):
            return "No dataset loaded."
        dataset_node = next((n for n in nodes.values() if isinstance(n, dict) and n.get("kind") == "dataset"), None)
        if not dataset_node:
            return "No dataset loaded."
        meta = dataset_node.get("metadata", {})
        dataset = meta.get("dataset", {}) if isinstance(meta, dict) else {}
        fields = dataset.get("fields", [])
        fields_text = ", ".join(fields) if isinstance(fields, list) and fields else "unknown"
        return (
            f"frames: {dataset.get('frames', 'unknown')} | "
            f"atoms: {dataset.get('atoms', 'unknown')} | "
            f"fields: {fields_text} | "
            f"engine: {dataset.get('engine_override') or dataset.get('engine_detected') or 'unknown'}"
        )

    @app.callback(
        Output("result-tab-content", "children"),
        Output("canvas-content", "children"),
        Input("result-tabs", "value"),
        Input("session-store", "data"),
        Input("result-store", "data"),
        Input("pipeline-store", "data"),
        Input("plot-x-col", "value"),
        Input("plot-y-col", "value"),
        Input("plot-group-col", "value"),
        Input("hist-col", "value"),
        Input("view3d-x-col", "value"),
        Input("view3d-y-col", "value"),
        Input("view3d-z-col", "value"),
        Input("view3d-color-col", "value"),
        Input("focus-atom", "value"),
    )
    def render_result_views(
        tab_node_id: str | None,
        session: dict[str, Any] | None,
        result_store: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
        x_col: str | None,
        y_col: str | None,
        group_col: str | None,
        hist_col: str | None,
        view3d_x: str | None,
        view3d_y: str | None,
        view3d_z: str | None,
        view3d_color: str | None,
        focus_atom: str | None,
    ):
        if not session:
            empty = "No selected node."
            return empty, empty
        node_id = str(session.get("selected_node_id") or "")
        nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
        selected_node = nodes.get(node_id) if isinstance(nodes, dict) else None
        if isinstance(selected_node, dict) and str(selected_node.get("kind")) == "analysis":
            empty = "No visualization selected. Select a visualization node under this analysis."
            return empty, empty
        if str(node_id).startswith("virtual:visualization:") and not tab_node_id:
            empty = "No presentations yet for this analysis."
            return empty, empty

        source_node_id = str(tab_node_id or node_id)
        source_node = nodes.get(source_node_id) if isinstance(nodes, dict) else None
        if not isinstance(source_node, dict):
            empty = "No visualization selected."
            return empty, empty
        presentation_spec = None
        if isinstance(source_node, dict):
            meta = source_node.get("metadata", {})
            if isinstance(meta, dict):
                presentation_spec = meta.get("presentation_spec")
        artifact = _find_source_artifact(snapshot, source_node_id, result_store or {})
        rows = _artifact_rows(artifact if isinstance(artifact, dict) else None)
        if focus_atom:
            rows = [r for r in rows if str(r.get("atom_id")) == str(focus_atom)]
        req = source_node.get("request", {}) if isinstance(source_node.get("request"), dict) else {}
        viz_type = str(req.get("visualization_type") or "plot2d").lower()

        if viz_type == "table":
            if not rows:
                content = "No result rows yet."
                return content, content
            table_filter_col = str(req.get("table_filter_col") or "")
            table_filter_value = str(req.get("table_filter_value") or "")
            table_max_rows = int(req.get("table_max_rows") or 200)
            if table_filter_col and table_filter_value:
                q = table_filter_value.strip().lower()
                if q:
                    rows = [r for r in rows if q in str(r.get(table_filter_col, "")).lower()]
            if not rows:
                content = "No rows match current filter."
                return content, content
            cols = [{"name": c, "id": c} for c in rows[0].keys()]
            table = dash_table.DataTable(
                data=rows[: max(10, min(10000, int(table_max_rows)))],
                columns=cols,
                page_size=15,
                style_table={"overflowX": "auto"},
                style_cell={"fontFamily": "Segoe UI", "fontSize": "12px", "textAlign": "left"},
            )
            return table, table

        if viz_type == "plot2d":
            use_x = str(req.get("x_col") or x_col or "iter")
            use_y = str(req.get("y_col") or y_col or "msd")
            use_group = str(req.get("group_col") or group_col or "")
            use_color = str(req.get("line_color_rgb") or req.get("line_color") or "")
            try:
                use_width = float(req.get("line_width")) if req.get("line_width") is not None else None
            except Exception:
                use_width = None
            fig = render_figure(
                rows,
                presentation=presentation_spec if isinstance(presentation_spec, dict) else None,
                x_col=use_x,
                y_col=use_y,
                group_col=use_group,
            )
            if fig is None:
                fig = _build_plot(
                    rows,
                    x_col=use_x,
                    y_col=use_y,
                    group_col=use_group,
                    line_color=use_color or None,
                    line_width=use_width,
                )
            else:
                for tr in fig.data:
                    if isinstance(tr, go.Scatter):
                        line_update: dict[str, Any] = {}
                        if use_color:
                            line_update["color"] = str(use_color)
                        if use_width is not None:
                            line_update["width"] = float(use_width)
                        if line_update:
                            tr.update(line=line_update)
            graph = dcc.Graph(figure=fig, config={"displaylogo": False})
            return graph, graph

        if viz_type == "histogram":
            use_hist = str(req.get("x_col") or req.get("y_col") or hist_col or "msd")
            fig = render_figure(
                rows,
                presentation=presentation_spec if isinstance(presentation_spec, dict) else None,
                x_col=use_hist,
                y_col=use_hist,
            )
            if fig is None:
                if not rows:
                    content = "No numeric data for histogram."
                    return content, content
                use_col = use_hist
                vals = []
                for row in rows:
                    val = row.get(use_col)
                    try:
                        vals.append(float(val))
                    except Exception:
                        continue
                if not vals:
                    content = "No numeric data for histogram."
                    return content, content
                fig = go.Figure(data=[go.Histogram(x=vals, nbinsx=40)])
                fig.update_layout(template="plotly_white", title=f"{use_col} Distribution")
            graph = dcc.Graph(figure=fig, config={"displaylogo": False})
            return graph, graph

        fig3d = render_figure(
            rows,
            presentation=presentation_spec if isinstance(presentation_spec, dict) else None,
            x_col=str(req.get("x_col") or view3d_x or ""),
            y_col=str(req.get("y_col") or view3d_y or ""),
            z_col=str(req.get("z_col") or view3d_z or ""),
            color_col=str(req.get("color_col") or view3d_color or ""),
        )
        if fig3d is None:
            fig3d = _build_3d(rows, x_col=view3d_x, y_col=view3d_y, z_col=view3d_z, color_col=view3d_color)
        graph3d = dcc.Graph(figure=fig3d, config={"displaylogo": False})
        return graph3d, graph3d
