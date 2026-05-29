"""Provide helper utilities for analysis callback registration.

This module contains the non-decorator helper logic used by
`reaxkit.webui.ui.analysis.callbacks`. It centralizes cache/state helpers,
table/plot shaping, and request normalization to keep callback registration
focused and readable.

**Usage context**

- Artifact/result cache lookup and normalization.
- Plot/table rendering helper functions for analysis views.
- Shared request/filter/style utilities used by callback handlers.
"""

from __future__ import annotations

import logging
from typing import Any
from numbers import Number
from datetime import date, datetime
import statistics
from hashlib import sha256

from dash import ALL, Input, Output, State, ctx, dash_table, dcc, html, no_update
import plotly.graph_objects as go
import plotly.io as pio
import os
import json
from pathlib import Path

from reaxkit.core.registry.analysis_cli_routing_registry import get_registered_analysis_commands
from reaxkit.core.registry.analysis_task_registry import TASK_LABELS, TASK_REGISTRY, task_display_label
from reaxkit.presentation.specs import ensure_presentation_spec, spec_to_dash_request
from reaxkit.presentation.export_utils import write_figure, write_table
from reaxkit.webui.backend.api import WebUIApiService
from reaxkit.webui.backend.tabular_payload import extract_tabular_rows, infer_columns, infer_numeric_columns
from reaxkit.webui.backend.utility_registry import canonical_utility_name, default_utility_request
from reaxkit.webui.presentation.perf_config import load_ui_performance_config
from reaxkit.webui.presentation.registry import render_figure
from reaxkit.webui.ui.analysis.tasks import (
    hidden_task_inputs,
    register_task_callbacks,
    render_task_analysis_properties,
)
from reaxkit.webui.ui.shell.callbacks import (
    default_workspace_dir_for_dataset as _default_workspace_dir_for_dataset,
)

logger = logging.getLogger(__name__)
_ARTIFACT_ROWS_CACHE: dict[str, list[dict[str, Any]]] = {}
_ARTIFACT_ROWS_CACHE_MAX = 128
_PLOT_ROWS_CACHE: dict[str, list[dict[str, Any]]] = {}
_PLOT_ROWS_CACHE_MAX = 96
_FIGURE_CACHE: dict[str, dict[str, Any]] = {}
_FIGURE_CACHE_MAX = 48
_ARTIFACT_OBJ_CACHE: dict[str, dict[str, Any]] = {}
_ARTIFACT_OBJ_CACHE_MAX = 96
_NODE_PIPELINE_CACHE: dict[str, str] = {}
_SERVICE_HANDLE: WebUIApiService | None = None
_UI_PERF = load_ui_performance_config()
_UI_PLOT2D_SCATTERGL_THRESHOLD = int(_UI_PERF["plot2d_scattergl_threshold"])
_UI_PLOT2D_MAX_POINTS = int(_UI_PERF["plot2d_max_points"])
_UI_PLOT2D_MIN_POINTS_PER_TRACE = int(_UI_PERF["plot2d_min_points_per_trace"])
_UI_PLOT2D_INITIAL_MAX_POINTS = int(_UI_PERF.get("plot2d_initial_max_points", 12000))
_UI_PLOT2D_ZOOM_MAX_POINTS = int(_UI_PERF.get("plot2d_zoom_max_points", 120000))
_UI_PLOT2D_MAX_CURVES_DISPLAY = int(_UI_PERF.get("plot2d_max_curves_display", 10))
if _UI_PLOT2D_ZOOM_MAX_POINTS < _UI_PLOT2D_INITIAL_MAX_POINTS:
    _UI_PLOT2D_ZOOM_MAX_POINTS = _UI_PLOT2D_INITIAL_MAX_POINTS


def _trace_enabled() -> bool:
    raw = str(os.environ.get("REAXKIT_UI_TRACE", "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _trace(message: str) -> None:
    if not _trace_enabled():
        return
    text = str(message)
    try:
        print(text, flush=True)
    except Exception:
        pass
    paths: list[Path] = []
    env_path = str(os.environ.get("REAXKIT_UI_TRACE_PATH", "")).strip()
    if env_path:
        paths.append(Path(env_path))
    else:
        paths.append(Path(_default_workspace_dir_for_dataset(str(Path.cwd()))) / "log" / "UI" / "ui_trace.log")
    for trace_path in paths:
        try:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        except Exception:
            continue
    try:
        logger.info(text)
    except Exception:
        pass


def _figure_cache_key(*, artifact_id: str, payload: dict[str, Any]) -> str | None:
    aid = str(artifact_id or "").strip()
    if not aid:
        return None
    try:
        text = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    except Exception:
        text = repr(payload)
    return f"{aid}:{sha256(text.encode('utf-8')).hexdigest()}"


def _figure_cache_get(key: str | None) -> dict[str, Any] | None:
    if not key:
        return None
    fig = _FIGURE_CACHE.get(key)
    if fig is None:
        return None
    # LRU touch
    _FIGURE_CACHE.pop(key, None)
    _FIGURE_CACHE[key] = fig
    return dict(fig)


def _figure_cache_put(key: str | None, figure: go.Figure | dict[str, Any] | None) -> None:
    if not key or figure is None:
        return
    try:
        fig_json = figure.to_plotly_json() if isinstance(figure, go.Figure) else dict(figure)
    except Exception:
        return
    _FIGURE_CACHE[key] = fig_json
    while len(_FIGURE_CACHE) > _FIGURE_CACHE_MAX:
        oldest = next(iter(_FIGURE_CACHE))
        _FIGURE_CACHE.pop(oldest, None)


def _artifact_cache_put(artifact: dict[str, Any] | None) -> None:
    if not isinstance(artifact, dict):
        return
    artifact_id = str(artifact.get("id") or "").strip()
    if not artifact_id:
        return
    _ARTIFACT_OBJ_CACHE[artifact_id] = artifact
    while len(_ARTIFACT_OBJ_CACHE) > _ARTIFACT_OBJ_CACHE_MAX:
        oldest = next(iter(_ARTIFACT_OBJ_CACHE))
        _ARTIFACT_OBJ_CACHE.pop(oldest, None)


def _artifact_id_from_entry(entry: Any) -> str:
    if isinstance(entry, str):
        return str(entry).strip()
    if isinstance(entry, dict):
        return str(entry.get("id") or "").strip()
    return ""


def _resolve_pipeline_id_for_node(node_id: str) -> str | None:
    nid = str(node_id or "").strip()
    if not nid:
        return None
    cached = _NODE_PIPELINE_CACHE.get(nid)
    if cached:
        return cached
    service = _SERVICE_HANDLE
    if service is None:
        return None
    store = getattr(service, "store", None)
    lock = getattr(store, "_lock", None)
    pipelines = getattr(store, "_pipelines", None)
    if not isinstance(pipelines, dict):
        return None
    try:
        cm = lock if lock is not None else None
        if cm is None:
            items = list(pipelines.items())
        else:
            with cm:
                items = list(pipelines.items())
    except Exception:
        items = list(pipelines.items())
    for pipeline_id, pipeline_state in items:
        nodes = getattr(pipeline_state, "nodes", {})
        if isinstance(nodes, dict) and nid in nodes:
            _NODE_PIPELINE_CACHE[nid] = str(pipeline_id)
            return str(pipeline_id)
    return None


def _resolve_artifact_by_id(artifact_id: str, *, node_id: str | None = None, pipeline_id: str | None = None) -> dict[str, Any] | None:
    aid = str(artifact_id or "").strip()
    if not aid:
        return None
    cached = _ARTIFACT_OBJ_CACHE.get(aid)
    if isinstance(cached, dict) and isinstance(cached.get("payload"), dict):
        _ARTIFACT_OBJ_CACHE.pop(aid, None)
        _ARTIFACT_OBJ_CACHE[aid] = cached
        return cached

    service = _SERVICE_HANDLE
    if service is None:
        return None
    store = getattr(service, "store", None)
    if store is None:
        return None

    pid = str(pipeline_id or "").strip()
    if not pid and node_id:
        pid = str(_resolve_pipeline_id_for_node(str(node_id)) or "").strip()

    artifact_obj = None
    if pid:
        try:
            artifact_obj = store.get_artifact(pid, aid)
        except Exception:
            artifact_obj = None

    if artifact_obj is None:
        lock = getattr(store, "_lock", None)
        pipelines = getattr(store, "_pipelines", None)
        if isinstance(pipelines, dict):
            try:
                cm = lock if lock is not None else None
                if cm is None:
                    items = list(pipelines.items())
                else:
                    with cm:
                        items = list(pipelines.items())
            except Exception:
                items = list(pipelines.items())
            for maybe_pid, pipeline_state in items:
                artifacts = getattr(pipeline_state, "artifacts", {})
                if isinstance(artifacts, dict) and aid in artifacts:
                    artifact_obj = artifacts.get(aid)
                    pid = str(maybe_pid)
                    break
        if artifact_obj is None:
            return None

    payload = getattr(artifact_obj, "payload", {})
    metadata = getattr(artifact_obj, "metadata", {})
    views = getattr(artifact_obj, "recommended_views", [])
    artifact_dict: dict[str, Any] = {
        "id": str(getattr(artifact_obj, "id", aid) or aid),
        "node_id": str(getattr(artifact_obj, "node_id", node_id or "") or ""),
        "payload": payload if isinstance(payload, dict) else {},
        "metadata": metadata if isinstance(metadata, dict) else {},
        "recommended_views": views if isinstance(views, list) else [],
        "created_at": str(getattr(artifact_obj, "created_at", "") or ""),
    }
    _artifact_cache_put(artifact_dict)
    node_key = str(artifact_dict.get("node_id") or "").strip()
    if pid and node_key:
        _NODE_PIPELINE_CACHE[node_key] = pid
    return artifact_dict


def _selected_node(snapshot: dict[str, Any] | None, session: dict[str, Any] | None) -> dict[str, Any] | None:
    if not snapshot or not session:
        return None
    node_id = session.get("selected_node_id")
    nodes = snapshot.get("nodes", {})
    if not isinstance(nodes, dict):
        return None
    node = nodes.get(node_id)
    return node if isinstance(node, dict) else None


def _snapshot_with_node_update(
    snapshot: dict[str, Any] | None,
    node_id: str,
    *,
    request: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    status: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(snapshot, dict):
        return snapshot
    nodes = snapshot.get("nodes", {})
    if not isinstance(nodes, dict):
        return snapshot
    old_node = nodes.get(str(node_id))
    if not isinstance(old_node, dict):
        return snapshot
    next_snapshot = dict(snapshot)
    next_nodes = dict(nodes)
    next_node = dict(old_node)
    if request is not None:
        next_node["request"] = dict(request)
    if metadata is not None:
        next_node["metadata"] = dict(metadata)
    if status is not None:
        next_node["status"] = str(status)
    next_nodes[str(node_id)] = next_node
    next_snapshot["nodes"] = next_nodes
    return next_snapshot


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
    artifact_id = str(artifact.get("id") or "").strip()
    if artifact_id and artifact_id in _ARTIFACT_ROWS_CACHE:
        return _ARTIFACT_ROWS_CACHE.get(artifact_id, [])
    payload = artifact.get("payload", {})
    rows: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        _trace(f"[UI_ERROR] _artifact_rows artifact_id={artifact.get('id')} payload is not a dict")
        logger.error("ui._artifact_rows artifact_id=%s payload is not a dict", artifact.get("id"))
        return []

    table = payload.get("table")
    if not isinstance(table, list):
        _trace(
            f"[UI_ERROR] _artifact_rows artifact_id={artifact.get('id')} missing/invalid payload['table']; payload_keys={sorted(payload.keys())}"
        )
        logger.error(
            "ui._artifact_rows artifact_id=%s missing/invalid payload['table']; payload_keys=%s",
            artifact.get("id"),
            sorted(payload.keys()),
        )
        return []

    rows = [dict(r) for r in table if isinstance(r, dict)]
    if not rows:
        _trace(f"[UI_ERROR] _artifact_rows artifact_id={artifact.get('id')} payload['table'] is empty or non-row")
        logger.error("ui._artifact_rows artifact_id=%s payload['table'] is empty or non-row", artifact.get("id"))
        return []

    if rows:
        logger.info("ui._artifact_rows artifact_id=%s rows=%s cols=%s", artifact.get("id"), len(rows), list(rows[0].keys()))
    else:
        logger.info("ui._artifact_rows artifact_id=%s rows=0 payload_keys=%s", artifact.get("id"), sorted(payload.keys()) if isinstance(payload, dict) else [])
    if artifact_id:
        _ARTIFACT_ROWS_CACHE[artifact_id] = rows
        while len(_ARTIFACT_ROWS_CACHE) > _ARTIFACT_ROWS_CACHE_MAX:
            oldest = next(iter(_ARTIFACT_ROWS_CACHE))
            _ARTIFACT_ROWS_CACHE.pop(oldest, None)
    return rows


def _build_result_table(rows: list[dict[str, Any]], *, max_rows: int = 200) -> dash_table.DataTable:
    safe_max = max(10, min(10000, int(max_rows)))
    column_ids = [str(c) for c in (rows[0].keys() if rows else [])]
    _trace(f"[UI_TRACE] _build_result_table rows={len(rows)} cols={column_ids} max_rows={safe_max}")
    normalized_rows: list[dict[str, Any]] = []
    for row in rows[:safe_max]:
        nr: dict[str, Any] = {}
        for key, val in row.items():
            cell = val
            if hasattr(cell, "item") and callable(getattr(cell, "item")):
                try:
                    cell = cell.item()
                except Exception:
                    pass
            if isinstance(cell, (dict, list, tuple, set)):
                cell = str(cell)
            nr[str(key)] = cell
        normalized_rows.append(nr)

    def _infer_col_type(values: list[Any]) -> str:
        saw_num = False
        saw_text = False
        saw_datetime = False
        for value in values:
            if value is None:
                continue
            if isinstance(value, bool):
                saw_text = True
            elif isinstance(value, Number):
                saw_num = True
            elif isinstance(value, (datetime, date)):
                saw_datetime = True
            else:
                saw_text = True
            if saw_text and (saw_num or saw_datetime):
                return "text"
        if saw_datetime and not saw_num and not saw_text:
            return "datetime"
        if saw_num and not saw_text and not saw_datetime:
            return "numeric"
        return "text"

    cols = []
    for col in column_ids:
        col_values = [r.get(col) for r in normalized_rows]
        cols.append({"name": col, "id": col, "type": _infer_col_type(col_values)})

    return dash_table.DataTable(
        data=normalized_rows,
        columns=cols,
        page_size=15,
        filter_action="native",
        filter_options={"case": "insensitive"},
        sort_action="native",
        sort_mode="multi",
        style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "Segoe UI", "fontSize": "12px", "textAlign": "left"},
    )


def _result_cache_from_snapshot(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    if not snapshot or not isinstance(snapshot, dict):
        return {}
    out: dict[str, Any] = {}
    nodes = snapshot.get("nodes", {})
    if isinstance(nodes, dict):
        for node in nodes.values():
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id") or "").strip()
            if not node_id:
                continue
            aid = str(node.get("result_ref") or "").strip()
            meta = node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {}
            if not aid:
                aid = str(meta.get("last_artifact_id") or "").strip()
            if aid:
                out[node_id] = aid
    if out:
        return out

    # Backward compatibility with snapshots that still include artifact maps.
    artifacts = snapshot.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return out
    for artifact in artifacts.values():
        if not isinstance(artifact, dict):
            continue
        node_id = str(artifact.get("node_id") or "").strip()
        aid = str(artifact.get("id") or "").strip()
        if node_id and aid:
            out[node_id] = aid
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
    if not isinstance(nodes, dict):
        return None
    cache = result_store or {}
    current_node = nodes.get(str(selected_node_id)) if isinstance(nodes, dict) else None

    def _resolve_for_node(node_id: str, expected_artifact_id: str | None = None) -> dict[str, Any] | None:
        entry = cache.get(node_id)
        if isinstance(entry, dict) and isinstance(entry.get("payload"), dict):
            if expected_artifact_id:
                if str(entry.get("id") or "").strip() == str(expected_artifact_id).strip():
                    _artifact_cache_put(entry)
                    return entry
            else:
                _artifact_cache_put(entry)
                return entry

        expected = str(expected_artifact_id or "").strip()
        entry_id = _artifact_id_from_entry(entry)
        if expected and entry_id and entry_id != expected:
            entry_id = expected
        if not entry_id:
            entry_id = expected
        if entry_id:
            resolved = _resolve_artifact_by_id(entry_id, node_id=node_id)
            if isinstance(resolved, dict):
                return resolved

        # Optional fallback for snapshots carrying inline artifacts.
        artifacts = snapshot.get("artifacts", {}) if isinstance(snapshot, dict) else {}
        if isinstance(artifacts, dict):
            if expected and isinstance(artifacts.get(expected), dict):
                candidate = artifacts.get(expected)
                if isinstance(candidate, dict) and isinstance(candidate.get("payload"), dict):
                    _artifact_cache_put(candidate)
                    return candidate
        return None

    # 1) direct cache
    if isinstance(current_node, dict):
        result_ref = str(current_node.get("result_ref") or "")
        meta = current_node.get("metadata", {})
        last_id = str(meta.get("last_artifact_id") or "") if isinstance(meta, dict) else ""
        direct = _resolve_for_node(str(selected_node_id), result_ref or last_id or None)
        if isinstance(direct, dict):
            return direct

    # 2) walk to nearest ancestor with artifact reference
    current = current_node
    seen: set[str] = set()
    while isinstance(current, dict):
        cid = str(current.get("id"))
        if cid in seen:
            break
        seen.add(cid)
        result_ref = current.get("result_ref")
        if result_ref:
            resolved = _resolve_for_node(cid, str(result_ref))
            if isinstance(resolved, dict):
                return resolved
        meta = current.get("metadata", {})
        if isinstance(meta, dict):
            last_id = meta.get("last_artifact_id")
            if last_id:
                resolved = _resolve_for_node(cid, str(last_id))
                if isinstance(resolved, dict):
                    return resolved
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


def _browse_save_file(
    *,
    title: str,
    initial_dir: Path,
    initial_name: str,
    filetypes: list[tuple[str, str]],
    default_ext: str,
) -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.asksaveasfilename(
            title=title,
            initialdir=str(initial_dir),
            initialfile=initial_name,
            defaultextension=default_ext,
            filetypes=filetypes,
        )
        root.destroy()
        if chosen and isinstance(chosen, str):
            return Path(chosen)
    except Exception:
        return None
    return None


def _default_export_dir(
    snapshot: dict[str, Any] | None,
    session: dict[str, Any] | None,
    config: dict[str, Any] | None,
) -> Path:
    cfg = dict(config or {})
    project_root = Path(
        str(cfg.get("workspace_dir") or _default_workspace_dir_for_dataset(str(Path.cwd())))
    ).resolve()
    dataset = _latest_dataset_node(snapshot)
    if isinstance(dataset, dict):
        meta = dataset.get("metadata", {}) if isinstance(dataset.get("metadata"), dict) else {}
        project_raw = str(meta.get("project_root") or "").strip()
        if project_raw:
            project_root = Path(project_raw).resolve()
    nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
    selected_node_id = str((session or {}).get("selected_node_id") or "")
    analysis_id = _ancestor_analysis_id(nodes, selected_node_id) if isinstance(nodes, dict) and selected_node_id else None
    workflow = "analysis"
    if analysis_id and isinstance(nodes, dict):
        anode = nodes.get(analysis_id)
        if isinstance(anode, dict):
            workflow = str(anode.get("metadata", {}).get("task_name") or anode.get("name") or "analysis")
    export_dir = project_root / "analysis" / workflow
    if analysis_id:
        export_dir = export_dir / str(analysis_id)
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def _resolve_canvas_source_node(
    snapshot: dict[str, Any] | None,
    session: dict[str, Any] | None,
    tab_node_id: str | None,
) -> dict[str, Any] | None:
    if not snapshot or not session:
        return None
    nodes = snapshot.get("nodes", {})
    if not isinstance(nodes, dict):
        return None
    node_id = str(session.get("selected_node_id") or "")
    selected_node = nodes.get(node_id)
    if isinstance(selected_node, dict) and str(selected_node.get("kind")) == "analysis":
        return None
    if str(node_id).startswith("virtual:visualization:") and not tab_node_id:
        return None
    source_node_id = str(tab_node_id or node_id)
    source_node = nodes.get(source_node_id)
    if not isinstance(source_node, dict):
        return None
    if str(source_node.get("kind")) == "utility":
        analysis_id = _ancestor_analysis_id(nodes, source_node_id)
        if analysis_id:
            viz_nodes = _visualization_nodes_for_analysis(snapshot, analysis_id)
            if viz_nodes:
                selected_viz = None
                for vnode in viz_nodes:
                    req = vnode.get("request", {}) if isinstance(vnode.get("request"), dict) else {}
                    meta = vnode.get("metadata", {}) if isinstance(vnode.get("metadata"), dict) else {}
                    vtype = _canonical_viz_type(req.get("visualization_type") or meta.get("visualization_type") or vnode.get("name"))
                    if vtype == "table":
                        selected_viz = vnode
                        break
                source_node = selected_viz or viz_nodes[0]
    return source_node if isinstance(source_node, dict) else None


def _engine_display_name(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if raw == "reaxff":
        return "ReaxFF"
    if raw == "lammps":
        return "LAMMPS"
    if raw == "ams":
        return "AMS"
    if raw == "autodetect":
        return "Autodetect"
    return str(value or "")


def _analysis_dropdown_options(
    task_names: list[str] | None,
    task_labels: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Build grouped analysis dropdown options."""
    tasks = sorted({str(task) for task in (task_names or []) if str(task).strip()})
    if not tasks:
        tasks = sorted(str(name) for name in TASK_REGISTRY.keys()) or sorted(
            str(name) for name in get_registered_analysis_commands().keys()
        )
    task_set = set(tasks)

    task_to_group: dict[str, str] = {}

    # Primary grouping source: analysis task registry module path
    # (e.g., reaxkit.analysis.timeseries.timeseries -> timeseries).
    for task_name in task_set:
        task_cls = TASK_REGISTRY.get(task_name)
        if task_cls is None:
            continue
        module_path = str(getattr(task_cls, "__module__", "")).strip()
        parts = [p for p in module_path.split(".") if p]
        if "analysis" in parts:
            i = parts.index("analysis")
            if i + 1 < len(parts):
                group_name = str(parts[i + 1]).strip().lower()
                if group_name:
                    task_to_group[task_name] = group_name

    # Fallback grouping source: CLI command routing registry.
    for task_name, spec in get_registered_analysis_commands().items():
        if task_name not in task_set or task_name in task_to_group:
            continue
        module_leaf = str(spec.module_path).split(".")[-1]
        group = module_leaf.replace("_workflow", "").strip().lower()
        if group:
            task_to_group[task_name] = group

    grouped: dict[str, list[str]] = {}
    for task_name in sorted(task_set):
        group = str(task_to_group.get(task_name) or "analysis")
        grouped.setdefault(group, []).append(str(task_name))

    options: list[dict[str, Any]] = []
    first_value = ""
    labels = {str(k): str(v) for k, v in (task_labels or {}).items()}
    for group_name in sorted(grouped.keys()):
        options.append(
            {
                "label": html.Span(group_name, style={"fontWeight": "700"}),
                "value": f"__group__:{group_name}",
                "disabled": True,
            }
        )
        for task_name in sorted(grouped[group_name]):
            task_label = labels.get(task_name)
            if not task_label:
                task_cls = TASK_REGISTRY.get(task_name)
                task_label = str(getattr(task_cls, "_reaxkit_task_label", "")).strip() if task_cls else ""
            if not task_label:
                task_label = str(TASK_LABELS.get(task_name) or task_display_label(task_name))
            options.append(
                {
                    "label": html.Span(task_label, style={"paddingLeft": "18px", "display": "inline-block"}),
                    "value": task_name,
                }
            )
            if not first_value:
                first_value = task_name

    if not options or not first_value:
        fallback = sorted(task_set)[0] if task_set else "msd"
        options = [{"label": fallback, "value": fallback}]
        first_value = fallback
    return options, first_value


def _catalog_payload(service: WebUIApiService) -> dict[str, Any]:
    try:
        catalog = service.get_catalog()
        return catalog if isinstance(catalog, dict) else {}
    except Exception:
        return {}


def _analysis_schema_map(catalog: dict[str, Any] | None) -> dict[str, Any]:
    raw = (catalog or {}).get("analysis_schemas", {})
    return raw if isinstance(raw, dict) else {}


def _analysis_defaults_for_task(task_name: str, catalog: dict[str, Any] | None) -> dict[str, Any]:
    schemas = _analysis_schema_map(catalog)
    schema = schemas.get(str(task_name), {})
    fields = schema.get("fields", []) if isinstance(schema, dict) else []
    out: dict[str, Any] = {}
    for field in fields:
        if not isinstance(field, dict):
            continue
        name = str(field.get("name") or "").strip()
        if not name:
            continue
        out[name] = field.get("default")
    return out


def _utility_specs_map(catalog: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    raw = (catalog or {}).get("utility_specs", [])
    if not isinstance(raw, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = canonical_utility_name(str(item.get("name") or ""))
        if name:
            out[name] = item
    return out


def _utility_field_options(
    utility_name: str,
    field_name: str,
    *,
    columns: list[str],
    numeric_columns: list[str],
) -> list[dict[str, str]]:
    name = str(field_name or "").strip().lower()
    util = canonical_utility_name(utility_name)
    choices: list[str] = []
    if name in {"column", "source"}:
        if util in {"denoise_ema", "denoise_sma", "column_transform"}:
            choices = list(numeric_columns or columns)
        else:
            choices = list(columns)
    elif name in {"group_by", "x_col"}:
        choices = list(columns)
    return [{"label": col, "value": col} for col in choices if str(col).strip()]


def _coerce_utility_field_value(raw_value: Any, field_type: str) -> Any:
    normalized = str(field_type or "Any").strip().lower()
    is_optional = "none" in normalized
    if raw_value is None:
        if "list[" in normalized:
            return []
        return None if is_optional else ""
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if text == "":
            if "list[" in normalized:
                return []
            return None if is_optional else ""
    else:
        text = str(raw_value).strip()

    if "list[int]" in normalized:
        if isinstance(raw_value, (list, tuple, set)):
            out: list[int] = []
            for item in raw_value:
                try:
                    out.append(int(item))
                except Exception:
                    continue
            return out
        return _parse_csv_ints(text) or []
    if "list[str]" in normalized:
        if isinstance(raw_value, (list, tuple, set)):
            return [str(item).strip() for item in raw_value if str(item).strip()]
        return _parse_csv_strs(text) or []
    if "bool" in normalized:
        if isinstance(raw_value, bool):
            return raw_value
        return text.lower() in {"1", "true", "yes", "on"}
    if "int" in normalized and "list[" not in normalized:
        try:
            return int(float(text))
        except Exception:
            return None if is_optional else 0
    if "float" in normalized and "list[" not in normalized:
        try:
            return float(text)
        except Exception:
            return None if is_optional else 0.0
    return str(raw_value)


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


def _latest_dataset_node(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
    if not isinstance(nodes, dict):
        return None
    datasets = [n for n in nodes.values() if isinstance(n, dict) and str(n.get("kind")) == "dataset"]
    if not datasets:
        return None
    datasets.sort(key=lambda n: str(n.get("updated_at", "")))
    return datasets[-1]


def _canonical_viz_type(raw: Any) -> str:
    val = str(raw or "").strip().lower().replace(" ", "")
    aliases = {
        "plot": "plot2d",
        "plot2d": "plot2d",
        "single_plot": "plot2d",
        "hist": "histogram",
        "histogram": "histogram",
        "scatter": "scatter3d",
        "scatter3d": "scatter3d",
        "3d": "scatter3d",
        "table": "table",
    }
    return aliases.get(val, val or "plot2d")


def _visualization_display_label(snapshot: dict[str, Any] | None, viz_node_id: str) -> str:
    nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
    if not isinstance(nodes, dict):
        return "visualization"
    node = nodes.get(str(viz_node_id))
    if not isinstance(node, dict):
        return "visualization"
    if str(node.get("kind")) != "visualization":
        return str(node.get("name") or "node")

    analysis_id = _ancestor_analysis_id(nodes, str(viz_node_id))
    if not analysis_id:
        raw = node.get("request", {}).get("visualization_type") if isinstance(node.get("request"), dict) else node.get("name")
        return _canonical_viz_type(raw)

    typed_counts: dict[str, int] = {}
    for vnode in _visualization_nodes_for_analysis(snapshot, analysis_id):
        current_id = str(vnode.get("id"))
        req = vnode.get("request", {}) if isinstance(vnode.get("request"), dict) else {}
        meta = vnode.get("metadata", {}) if isinstance(vnode.get("metadata"), dict) else {}
        raw_type = req.get("visualization_type") or meta.get("visualization_type") or vnode.get("name")
        vtype = _canonical_viz_type(raw_type)
        typed_counts[vtype] = typed_counts.get(vtype, 0) + 1
        if current_id == str(viz_node_id):
            return f"{vtype}: {typed_counts[vtype]:02d}"
    return str(node.get("name") or "visualization")


def _analysis_display_label(snapshot: dict[str, Any] | None, analysis_node_id: str) -> str:
    nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
    if not isinstance(nodes, dict):
        return "analysis"
    node = nodes.get(str(analysis_node_id))
    if not isinstance(node, dict):
        return "analysis"
    task_key = str(node.get("metadata", {}).get("task_name") or node.get("name") or "analysis").strip().lower()
    raw_name = str(TASK_LABELS.get(task_key) or task_display_label(task_key))
    key = task_key
    analyses = [
        n for n in nodes.values()
        if isinstance(n, dict) and str(n.get("kind")) == "analysis"
    ]
    analyses.sort(key=lambda n: str(n.get("created_at", "")))
    idx = 0
    for item in analyses:
        item_name = str(item.get("metadata", {}).get("task_name") or item.get("name") or "analysis").strip().lower()
        if item_name == key:
            idx += 1
        if str(item.get("id")) == str(analysis_node_id):
            return f"{raw_name}: {idx:02d}"
    return raw_name


def _utility_display_label(snapshot: dict[str, Any] | None, utility_node_id: str) -> str:
    nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
    if not isinstance(nodes, dict):
        return "utility"
    node = nodes.get(str(utility_node_id))
    if not isinstance(node, dict):
        return "utility"
    raw_name = str(node.get("metadata", {}).get("utility_name") or node.get("name") or "utility")
    key = raw_name.strip().lower()
    utilities = [
        n for n in nodes.values()
        if isinstance(n, dict) and str(n.get("kind")) == "utility"
    ]
    utilities.sort(key=lambda n: str(n.get("created_at", "")))
    idx = 0
    for item in utilities:
        item_name = str(item.get("metadata", {}).get("utility_name") or item.get("name") or "utility").strip().lower()
        if item_name == key:
            idx += 1
        if str(item.get("id")) == str(utility_node_id):
            return f"{raw_name}: {idx:02d}"
    return raw_name


def _node_display_label(snapshot: dict[str, Any] | None, node_id: str) -> str:
    nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
    if not isinstance(nodes, dict):
        return node_id
    node = nodes.get(str(node_id))
    if not isinstance(node, dict):
        return node_id
    kind = str(node.get("kind") or "")
    if kind == "analysis":
        return _analysis_display_label(snapshot, str(node_id))
    if kind == "utility":
        return _utility_display_label(snapshot, str(node_id))
    return str(node.get("name") or node_id)


def _node_has_materialized_result(node: dict[str, Any] | None) -> bool:
    if not isinstance(node, dict):
        return False
    if node.get("result_ref"):
        return True
    meta = node.get("metadata", {})
    return isinstance(meta, dict) and bool(meta.get("last_artifact_id"))


def _join_source_options(
    snapshot: dict[str, Any] | None,
    *,
    exclude_node_ids: set[str] | None = None,
) -> list[dict[str, str]]:
    nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
    if not isinstance(nodes, dict):
        return []
    excluded = {str(v) for v in (exclude_node_ids or set()) if str(v).strip()}
    items: list[dict[str, str]] = []
    for node in nodes.values():
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "")
        if not node_id or node_id in excluded:
            continue
        kind = str(node.get("kind") or "")
        if kind not in {"analysis", "utility"}:
            continue
        if not _node_has_materialized_result(node):
            continue
        items.append({"label": _node_display_label(snapshot, node_id), "value": node_id})
    items.sort(key=lambda item: (item["label"].lower(), item["value"]))
    return items


def _join_key_candidates(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]) -> list[str]:
    left_cols = infer_columns(left_rows)
    right_cols = set(infer_columns(right_rows))
    shared = [col for col in left_cols if col in right_cols]
    preferred = [col for col in ("frame_index", "iter", "atom_id", "src", "dst") if col in shared]
    tail = [col for col in shared if col not in preferred]
    return preferred + tail


def _normalize_join_keys(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [tok.strip() for tok in raw.split(",") if tok.strip()]
    if isinstance(raw, (list, tuple, set)):
        return [str(tok).strip() for tok in raw if str(tok).strip()]
    return []


def _viz_request_with_defaults(existing: dict[str, Any] | None, viz_type: str) -> dict[str, Any]:
    req = dict(existing or {})
    req["visualization_type"] = str(viz_type)
    req.setdefault("x_col", "")
    req.setdefault("y_col", "")
    req.setdefault("z_col", "")
    req.setdefault("use_plot_title", False)
    req.setdefault("plot_title", "")
    req.setdefault("x_title", "")
    req.setdefault("y_title", "")
    req.setdefault("z_title", "")
    req.setdefault("color_col", "")
    req.setdefault("group_col", "")
    req.setdefault("group_agg", "none")
    req.setdefault("row_filters", [])
    req.setdefault("line_color", "blue")
    req.setdefault("line_color_rgb", "")
    req.setdefault("line_width", 2.0)
    req.setdefault("table_filter_col", "")
    req.setdefault("table_filter_value", "")
    req.setdefault("table_max_rows", 200)
    req.setdefault("font_size", 12)
    req.setdefault("marker_size", 0 if str(viz_type).lower() == "plot2d" else 6)
    req.setdefault("theme", "plotly_white")
    req.setdefault("axis_title_size", 13)
    req.setdefault("grid_on", True)
    req.setdefault("axis_box_on", True)
    req.setdefault("axis_box_width", 1.5)
    req.setdefault("log_scale", "none")
    req.setdefault("tick_spacing_x", "")
    req.setdefault("tick_spacing_y", "")
    req.setdefault("legend_position", "top-right")
    req.setdefault("show_legend", True)
    req.setdefault("show_markers", False)
    req.setdefault("trace_styles", {})
    return req


def _parse_float(raw: Any, default: float | None = None) -> float | None:
    if raw is None:
        return default
    txt = str(raw).strip()
    if txt == "":
        return default
    try:
        return float(txt)
    except Exception:
        return default


_ROW_FILTER_OPERATORS: tuple[str, ...] = (
    "==",
    "!=",
    ">",
    ">=",
    "<",
    "<=",
    "in",
    "not in",
    "contains",
    "between",
)


def _row_filters_from_raw(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        column = str(item.get("column") or "").strip()
        op = str(item.get("op") or "==").strip().lower()
        value = str(item.get("value") or "").strip()
        if not column:
            continue
        if op not in _ROW_FILTER_OPERATORS:
            op = "=="
        out.append({"column": column, "op": op, "value": value})
    return out


def _row_filter_table_data(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return [{"column": "", "op": "==", "value": ""}]
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        column = str(item.get("column") or "").strip()
        op = str(item.get("op") or "==").strip().lower()
        value = str(item.get("value") or "")
        if op not in _ROW_FILTER_OPERATORS:
            op = "=="
        out.append({"column": column, "op": op, "value": value})
    return out or [{"column": "", "op": "==", "value": ""}]


def _row_filter_dropdown(columns: list[str] | None) -> dict[str, Any]:
    col_opts = [{"label": str(col), "value": str(col)} for col in (columns or []) if str(col).strip()]
    return {
        "column": {"options": col_opts},
        "op": {"options": [{"label": op, "value": op} for op in _ROW_FILTER_OPERATORS]},
    }


def _as_num(value: Any) -> float | None:
    return _parse_float(value, None)


def _compare_row_filter(row_value: Any, op: str, filter_value: str) -> bool:
    op_norm = str(op or "==").strip().lower()
    left_num = _as_num(row_value)
    left_text = str(row_value or "").strip()
    raw = str(filter_value or "").strip()

    if op_norm in {"in", "not in"}:
        tokens = [tok.strip() for tok in raw.split(",") if tok.strip()]
        hit = left_text in tokens
        return (not hit) if op_norm == "not in" else hit
    if op_norm == "contains":
        return raw.lower() in left_text.lower()
    if op_norm == "between":
        parts = [tok.strip() for tok in raw.split(",") if tok.strip()]
        if len(parts) != 2:
            return True
        lo_num = _as_num(parts[0])
        hi_num = _as_num(parts[1])
        if left_num is not None and lo_num is not None and hi_num is not None:
            lo = min(lo_num, hi_num)
            hi = max(lo_num, hi_num)
            return lo <= left_num <= hi
        lo_txt, hi_txt = sorted(parts)
        return lo_txt <= left_text <= hi_txt

    right_num = _as_num(raw)
    if left_num is not None and right_num is not None:
        if op_norm == "==":
            return left_num == right_num
        if op_norm == "!=":
            return left_num != right_num
        if op_norm == ">":
            return left_num > right_num
        if op_norm == ">=":
            return left_num >= right_num
        if op_norm == "<":
            return left_num < right_num
        if op_norm == "<=":
            return left_num <= right_num
        return True

    if op_norm == "==":
        return left_text == raw
    if op_norm == "!=":
        return left_text != raw
    if op_norm == ">":
        return left_text > raw
    if op_norm == ">=":
        return left_text >= raw
    if op_norm == "<":
        return left_text < raw
    if op_norm == "<=":
        return left_text <= raw
    return True


def _apply_row_filters(rows: list[dict[str, Any]], raw_filters: Any) -> list[dict[str, Any]]:
    filters = _row_filters_from_raw(raw_filters)
    if not filters:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        keep = True
        for fil in filters:
            col = fil["column"]
            if col not in row:
                continue
            if not _compare_row_filter(row.get(col), fil["op"], fil["value"]):
                keep = False
                break
        if keep:
            out.append(row)
    return out


def _trace_styles_map(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, style in raw.items():
        if not isinstance(style, dict):
            continue
        skey = str(key).strip()
        if not skey:
            continue
        out[skey] = dict(style)
    return out


def _trace_key_candidates(trace_name: str, curve_index: int) -> list[str]:
    clean_name = str(trace_name or "").strip()
    return [clean_name, str(curve_index), f"curve_{curve_index}"]


def _trace_style_for_trace(trace_styles: dict[str, dict[str, Any]], trace_name: str, curve_index: int) -> dict[str, Any]:
    for key in _trace_key_candidates(trace_name, curve_index):
        if key and key in trace_styles:
            style = trace_styles.get(key)
            if isinstance(style, dict):
                return style
    return {}


def _normalize_group_agg(raw: Any) -> str:
    val = str(raw or "none").strip().lower()
    allowed = {"none", "mean", "median", "min", "max", "sum", "count", "std"}
    return val if val in allowed else "none"


def _aggregate_plot2d_rows(
    rows: list[dict[str, Any]],
    *,
    x_col: str,
    y_col: str,
    group_col: str,
    agg: str,
) -> list[dict[str, Any]]:
    mode = _normalize_group_agg(agg)
    if mode == "none" or not rows:
        return rows

    buckets: dict[tuple[str, Any], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        xv = row.get(x_col)
        yv = row.get(y_col)
        if xv is None or yv is None:
            continue
        try:
            y_num = float(yv)
        except Exception:
            continue
        gkey = str(row.get(group_col, "all")) if group_col else "all"
        key = (gkey, xv)
        bucket = buckets.setdefault(key, {"group": gkey, "x": xv, "vals": []})
        bucket["vals"].append(y_num)

    out: list[dict[str, Any]] = []
    for bucket in buckets.values():
        vals = list(bucket["vals"])
        if not vals:
            continue
        if mode == "mean":
            y_out = float(sum(vals) / len(vals))
        elif mode == "median":
            y_out = float(statistics.median(vals))
        elif mode == "min":
            y_out = float(min(vals))
        elif mode == "max":
            y_out = float(max(vals))
        elif mode == "sum":
            y_out = float(sum(vals))
        elif mode == "count":
            y_out = float(len(vals))
        elif mode == "std":
            y_out = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
        else:
            y_out = float(sum(vals) / len(vals))
        row_out: dict[str, Any] = {x_col: bucket["x"], y_col: y_out}
        if group_col:
            row_out[group_col] = bucket["group"]
        out.append(row_out)
    return out


def _cached_aggregate_plot2d_rows(
    rows: list[dict[str, Any]],
    *,
    artifact_id: str,
    x_col: str,
    y_col: str,
    group_col: str,
    agg: str,
    filter_signature: str,
) -> list[dict[str, Any]]:
    mode = _normalize_group_agg(agg)
    if mode == "none" or not rows or not artifact_id:
        return _aggregate_plot2d_rows(rows, x_col=x_col, y_col=y_col, group_col=group_col, agg=agg)
    key = "|".join(
        [
            str(artifact_id),
            str(x_col),
            str(y_col),
            str(group_col),
            str(mode),
            str(filter_signature),
            str(len(rows)),
        ]
    )
    cached = _PLOT_ROWS_CACHE.get(key)
    if isinstance(cached, list):
        return cached
    out = _aggregate_plot2d_rows(rows, x_col=x_col, y_col=y_col, group_col=group_col, agg=agg)
    _PLOT_ROWS_CACHE[key] = out
    while len(_PLOT_ROWS_CACHE) > _PLOT_ROWS_CACHE_MAX:
        oldest = next(iter(_PLOT_ROWS_CACHE))
        _PLOT_ROWS_CACHE.pop(oldest, None)
    return out


def _series_len(values: Any) -> int:
    try:
        return len(values) if values is not None else 0
    except Exception:
        return 0


def _downsample_indices(total: int, max_points: int) -> list[int]:
    if total <= 0 or max_points <= 0 or total <= max_points:
        return list(range(total))
    step = max(1, (total + max_points - 1) // max_points)
    idx = list(range(0, total, step))
    if not idx or idx[-1] != total - 1:
        idx.append(total - 1)
    if len(idx) > max_points:
        stride = max(1, (len(idx) + max_points - 1) // max_points)
        idx = idx[::stride]
        if idx[-1] != total - 1:
            idx.append(total - 1)
    return idx


def _take_by_index(values: Any, indices: list[int]) -> list[Any]:
    out: list[Any] = []
    if values is None:
        return out
    for i in indices:
        try:
            out.append(values[i])
        except Exception:
            break
    return out


def _downsample_scatter_payload(payload: dict[str, Any], *, max_points: int) -> dict[str, Any]:
    x_vals = payload.get("x")
    y_vals = payload.get("y")
    total = min(_series_len(x_vals), _series_len(y_vals))
    if total <= 0 or total <= int(max_points):
        return payload
    idx = _downsample_indices(total, int(max_points))
    if not idx:
        return payload
    payload["x"] = _take_by_index(x_vals, idx)
    payload["y"] = _take_by_index(y_vals, idx)

    for key in ("text", "hovertext", "ids", "customdata"):
        raw = payload.get(key)
        if _series_len(raw) == total:
            payload[key] = _take_by_index(raw, idx)
    marker = payload.get("marker")
    if isinstance(marker, dict):
        for mkey in ("color", "size", "opacity"):
            raw = marker.get(mkey)
            if _series_len(raw) == total:
                marker[mkey] = _take_by_index(raw, idx)
    return payload


def _optimize_plot2d_for_large_data(
    fig: go.Figure,
    *,
    threshold_points: int | None = None,
    max_points_total: int | None = None,
) -> go.Figure:
    scatter_traces: list[tuple[go.Scatter, int]] = []
    point_count = 0
    for tr in fig.data:
        if isinstance(tr, go.Scatter):
            n = _series_len(getattr(tr, "x", None))
            scatter_traces.append((tr, n))
            point_count += n
    if not scatter_traces:
        return fig

    threshold = int(threshold_points or _UI_PLOT2D_SCATTERGL_THRESHOLD)
    max_total = int(max_points_total or _UI_PLOT2D_MAX_POINTS)
    min_per_trace = int(_UI_PLOT2D_MIN_POINTS_PER_TRACE)
    needs_scattergl = point_count > threshold
    needs_downsample = point_count > max_total and max_total > 0
    if not needs_scattergl and not needs_downsample:
        return fig

    # Compute per-trace target counts when total points are too high.
    per_trace_limit: dict[int, int] = {}
    if needs_downsample and point_count > 0:
        ratio = float(max_total) / float(point_count)
        for idx, (_, npts) in enumerate(scatter_traces):
            keep = int(npts * ratio)
            keep = max(min_per_trace, keep)
            keep = min(npts, keep)
            per_trace_limit[idx] = keep

    new_fig = go.Figure()
    new_fig.update_layout(fig.layout)
    scatter_idx = -1
    for tr in fig.data:
        if not isinstance(tr, go.Scatter):
            new_fig.add_trace(tr)
            continue
        scatter_idx += 1
        payload = tr.to_plotly_json()
        payload.pop("type", None)
        if needs_downsample:
            limit = per_trace_limit.get(scatter_idx, _series_len(payload.get("x")))
            payload = _downsample_scatter_payload(payload, max_points=limit)
        new_fig.add_trace(go.Scattergl(**payload) if needs_scattergl else go.Scatter(**payload))
    return new_fig


def _cap_plot2d_curve_count(fig: go.Figure, *, max_curves: int) -> go.Figure:
    limit = max(1, int(max_curves or 1))
    curves_total = 0
    kept: list[Any] = []
    curves_kept = 0
    for tr in fig.data:
        is_curve = isinstance(tr, (go.Scatter, go.Scattergl))
        if is_curve:
            curves_total += 1
            if curves_kept >= limit:
                continue
            curves_kept += 1
        kept.append(tr)
    if curves_total <= limit:
        return fig

    capped = go.Figure()
    capped.update_layout(fig.layout)
    for tr in kept:
        capped.add_trace(tr)
    message = f"Showing {curves_kept} of {curves_total} atom curves."
    meta_raw = capped.layout.meta
    meta: dict[str, Any] = dict(meta_raw) if isinstance(meta_raw, dict) else {}
    meta.update(
        {
            "curve_cap_applied": True,
            "curve_cap_total": curves_total,
            "curve_cap_shown": curves_kept,
            "curve_cap_status": message,
        }
    )
    capped.update_layout(meta=meta)
    return capped


def _curve_cap_status_from_figure(fig: go.Figure | None) -> str:
    if not isinstance(fig, go.Figure):
        return ""
    meta_raw = fig.layout.meta
    if not isinstance(meta_raw, dict):
        return ""
    return str(meta_raw.get("curve_cap_status") or "").strip()


def _extract_relayout_x_range(relayout: dict[str, Any] | None) -> tuple[tuple[float, float] | None, bool]:
    if not isinstance(relayout, dict) or not relayout:
        return None, False
    if bool(relayout.get("xaxis.autorange")):
        return None, True
    low_raw = relayout.get("xaxis.range[0]")
    high_raw = relayout.get("xaxis.range[1]")
    if low_raw is None or high_raw is None:
        xr = relayout.get("xaxis.range")
        if isinstance(xr, (list, tuple)) and len(xr) >= 2:
            low_raw, high_raw = xr[0], xr[1]
    low = _parse_float(low_raw, None)
    high = _parse_float(high_raw, None)
    if low is None or high is None:
        return None, False
    lo, hi = (float(low), float(high))
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi), True


def _rows_in_numeric_x_range(
    rows: list[dict[str, Any]],
    *,
    x_col: str,
    x_range: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    if not rows or not x_col or x_range is None:
        return rows
    lo, hi = x_range
    out: list[dict[str, Any]] = []
    parsed_numeric = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        xv = _parse_float(row.get(x_col), None)
        if xv is None:
            continue
        parsed_numeric += 1
        if lo <= float(xv) <= hi:
            out.append(row)
    if parsed_numeric == 0:
        # Non-numeric x-axis; fall back to unfiltered rows.
        return rows
    return out


def _build_plot2d_figure(
    *,
    rows: list[dict[str, Any]],
    artifact_id: str,
    presentation_spec: dict[str, Any] | None,
    req: dict[str, Any],
    x_col: str | None,
    y_col: str | None,
    group_col: str | None,
    x_range: tuple[float, float] | None = None,
    max_points_total: int | None = None,
    cache_scope: str = "default",
) -> go.Figure | None:
    cols = infer_columns(rows)
    numeric_cols = infer_numeric_columns(rows)
    fallback_x = "iter" if "iter" in cols else ("frame_index" if "frame_index" in cols else (cols[0] if cols else ""))
    fallback_y = next((col for col in numeric_cols if col != fallback_x), numeric_cols[0] if numeric_cols else (cols[1] if len(cols) > 1 else fallback_x))
    use_x = str(req.get("x_col") or x_col or fallback_x)
    use_y = str(req.get("y_col") or y_col or fallback_y)
    use_group = str(req.get("group_col") or group_col or "")
    use_group_agg = _normalize_group_agg(req.get("group_agg"))
    use_color = str(req.get("line_color_rgb") or req.get("line_color") or "")
    try:
        use_width = float(req.get("line_width")) if req.get("line_width") is not None else None
    except Exception:
        use_width = None
    use_marker = _parse_float(req.get("marker_size"), 6.0)
    show_markers = _flag_on(req.get("show_markers"), default=False)
    trace_styles = _trace_styles_map(req.get("trace_styles"))
    row_filters = _row_filters_from_raw(req.get("row_filters"))
    x_range_sig = [round(x_range[0], 9), round(x_range[1], 9)] if x_range is not None else None
    fig_cache_payload = {
        "viz_type": "plot2d",
        "x": use_x,
        "y": use_y,
        "group": use_group,
        "group_agg": use_group_agg,
        "row_filters": row_filters,
        "max_curves_display": int(_UI_PLOT2D_MAX_CURVES_DISPLAY),
        "line_color": use_color,
        "line_width": use_width,
        "marker_size": use_marker,
        "show_markers": show_markers,
        "trace_styles": trace_styles,
        "x_range": x_range_sig,
        "max_points_total": int(max_points_total or _UI_PLOT2D_MAX_POINTS),
        "cache_scope": str(cache_scope),
        "style": {
            "theme": str(req.get("theme") or "plotly_white"),
            "font_size": _parse_float(req.get("font_size"), 12.0),
            "axis_title_size": _parse_float(req.get("axis_title_size"), 13.0),
            "grid_on": _flag_on(req.get("grid_on"), default=True),
            "axis_box_on": _flag_on(req.get("axis_box_on"), default=True),
            "axis_box_width": _parse_float(req.get("axis_box_width"), 1.5),
            "log_scale": str(req.get("log_scale") or "none"),
            "tick_spacing_x": _parse_float(req.get("tick_spacing_x"), None),
            "tick_spacing_y": _parse_float(req.get("tick_spacing_y"), None),
            "show_legend": _flag_on(req.get("show_legend"), default=True),
            "legend_position": str(req.get("legend_position") or "top-right"),
            "use_plot_title": _flag_on(req.get("use_plot_title"), default=False),
            "plot_title": str(req.get("plot_title") or "").strip(),
            "x_title": str(req.get("x_title") or "").strip(),
            "y_title": str(req.get("y_title") or "").strip(),
        },
    }
    fig_cache_key = _figure_cache_key(artifact_id=artifact_id, payload=fig_cache_payload)
    cached_fig_json = _figure_cache_get(fig_cache_key)
    if isinstance(cached_fig_json, dict):
        return go.Figure(cached_fig_json)

    rows_for_view = _rows_in_numeric_x_range(rows, x_col=use_x, x_range=x_range)
    if not rows_for_view:
        return None

    row_filter_signature = json.dumps({"row_filters": row_filters, "x_range": x_range_sig}, sort_keys=True, ensure_ascii=True)
    plot_rows = _cached_aggregate_plot2d_rows(
        rows_for_view,
        artifact_id=artifact_id,
        x_col=use_x,
        y_col=use_y,
        group_col=use_group,
        agg=use_group_agg,
        filter_signature=row_filter_signature,
    )
    fig = render_figure(
        plot_rows,
        presentation=presentation_spec if isinstance(presentation_spec, dict) else None,
        x_col=use_x,
        y_col=use_y,
        group_col=use_group,
        view_type="plot2d",
    )
    if fig is None:
        return None
    fig = _cap_plot2d_curve_count(fig, max_curves=_UI_PLOT2D_MAX_CURVES_DISPLAY)

    scatter_count = sum(1 for tr in fig.data if isinstance(tr, go.Scatter))
    apply_fixed_line_color = bool(use_color) and scatter_count <= 1 and not bool(trace_styles)
    for curve_index, tr in enumerate(fig.data):
        if not isinstance(tr, go.Scatter):
            continue
        trace_name = str(tr.name or f"curve_{curve_index}")
        curve_style = _trace_style_for_trace(trace_styles, trace_name, curve_index)
        curve_color = str(curve_style.get("line_color_rgb") or curve_style.get("line_color") or "").strip()
        curve_width = _parse_float(curve_style.get("line_width"), None)
        line_update: dict[str, Any] = {}
        if curve_color:
            line_update["color"] = curve_color
        elif apply_fixed_line_color:
            line_update["color"] = str(use_color)
        if curve_width is not None:
            line_update["width"] = float(curve_width)
        elif use_width is not None:
            line_update["width"] = float(use_width)
        if line_update:
            tr.update(line=line_update)

    for curve_index, tr in enumerate(fig.data):
        if not isinstance(tr, go.Scatter):
            continue
        trace_name = str(tr.name or f"curve_{curve_index}")
        curve_style = _trace_style_for_trace(trace_styles, trace_name, curve_index)
        curve_marker_size = _parse_float(curve_style.get("marker_size"), None)
        has_curve_marker_flag = "show_markers" in curve_style
        curve_show_markers = _flag_on(curve_style.get("show_markers"), default=False) if has_curve_marker_flag else show_markers
        mode = str(tr.mode or "lines")
        marker_size = curve_marker_size if curve_marker_size is not None else use_marker
        if curve_show_markers and marker_size is not None and marker_size > 0:
            if "markers" not in mode:
                mode = f"{mode}+markers" if mode else "markers"
            tr.update(mode=mode, marker={"size": float(marker_size)})
        else:
            mode_clean = mode.replace("+markers", "").replace("markers+", "")
            tr.update(mode=(mode_clean if mode_clean else "lines"))

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

    fig = _optimize_plot2d_for_large_data(fig, max_points_total=max_points_total)
    fig.update_layout(
        autosize=True,
        height=None,
        uirevision=f"rk-plot2d:{artifact_id}:{use_x}:{use_y}:{use_group}",
    )
    if x_range is not None:
        fig.update_xaxes(range=[x_range[0], x_range[1]], autorange=False)

    _figure_cache_put(fig_cache_key, fig)
    return fig


def _prime_rows_cache_from_artifact(artifact: dict[str, Any] | None) -> int:
    if not isinstance(artifact, dict):
        return 0
    _artifact_cache_put(artifact)
    artifact_id = str(artifact.get("id") or "").strip()
    payload = artifact.get("payload", {})
    table = payload.get("table") if isinstance(payload, dict) else None
    if not artifact_id or not isinstance(table, list):
        return 0
    rows = [dict(row) for row in table if isinstance(row, dict)]
    if not rows:
        return 0
    _ARTIFACT_ROWS_CACHE[artifact_id] = rows
    while len(_ARTIFACT_ROWS_CACHE) > _ARTIFACT_ROWS_CACHE_MAX:
        oldest = next(iter(_ARTIFACT_ROWS_CACHE))
        _ARTIFACT_ROWS_CACHE.pop(oldest, None)
    return len(rows)


def _precompute_plot2d_cache_for_analysis(
    *,
    snapshot: dict[str, Any] | None,
    analysis_id: str | None,
    result_store: dict[str, Any] | None,
    max_plot_nodes: int = 6,
) -> int:
    aid = str(analysis_id or "").strip()
    if not snapshot or not aid:
        return 0
    viz_nodes = _visualization_nodes_for_analysis(snapshot, aid)
    if not viz_nodes:
        return 0

    primed = 0
    for vnode in viz_nodes:
        if primed >= int(max_plot_nodes):
            break
        if not isinstance(vnode, dict):
            continue
        req = vnode.get("request", {}) if isinstance(vnode.get("request"), dict) else {}
        meta = vnode.get("metadata", {}) if isinstance(vnode.get("metadata"), dict) else {}
        vtype = _canonical_viz_type(req.get("visualization_type") or meta.get("visualization_type") or vnode.get("name"))
        if vtype != "plot2d":
            continue

        source_node_id = str(vnode.get("id") or "").strip()
        if not source_node_id:
            continue
        artifact = _find_source_artifact(snapshot, source_node_id, result_store or {})
        if not isinstance(artifact, dict):
            continue
        _prime_rows_cache_from_artifact(artifact)
        rows = _artifact_rows(artifact)
        if not rows:
            continue
        rows = _apply_row_filters(rows, req.get("row_filters"))
        if not rows:
            continue

        artifact_id = str(artifact.get("id") or "").strip()
        presentation_spec = meta.get("presentation_spec") if isinstance(meta, dict) else None
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
            continue
        primed += 1
    return primed


def _theme_options() -> list[dict[str, str]]:
    cached = getattr(_theme_options, "_cache", None)
    if isinstance(cached, list) and cached:
        return cached
    built_in = ["plotly_white", "plotly", "plotly_dark", "simple_white", "ggplot2", "seaborn", "presentation"]
    bootstrap_like = [
        "bootstrap",
        "cerulean",
        "cosmo",
        "cyborg",
        "darkly",
        "flatly",
        "journal",
        "litera",
        "lumen",
        "lux",
        "materia",
        "minty",
        "morph",
        "pulse",
        "quartz",
        "sandstone",
        "simplex",
        "sketchy",
        "slate",
        "solar",
        "spacelab",
        "superhero",
        "united",
        "vapor",
        "yeti",
        "zephyr",
    ]
    # Optional bridge package: registers Bootstrap-like Plotly templates.
    try:
        from dash_bootstrap_templates import load_figure_template

        load_figure_template(bootstrap_like)
    except Exception:
        pass

    names: list[str] = []
    for t in built_in + bootstrap_like:
        if t in pio.templates and t not in names:
            names.append(t)
    opts = [{"label": n, "value": n} for n in names]
    setattr(_theme_options, "_cache", opts)
    return opts


def _safe_theme(theme: Any) -> str:
    candidate = str(theme or "plotly_white").strip()
    return candidate if candidate in pio.templates else "plotly_white"


def _flag_on(raw: Any, default: bool = True) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, list):
        return any(str(v).strip().lower() == "on" for v in raw)
    txt = str(raw).strip().lower()
    if txt in {"on", "true", "1", "yes"}:
        return True
    if txt in {"off", "false", "0", "no"}:
        return False
    return default


def _legend_layout(position: str | None) -> dict[str, Any]:
    pos = str(position or "top-right").strip().lower()
    if pos == "hidden":
        return {"showlegend": False}
    mapping = {
        "top-right": {"x": 1.0, "y": 1.0, "xanchor": "right", "yanchor": "top"},
        "top-left": {"x": 0.0, "y": 1.0, "xanchor": "left", "yanchor": "top"},
        "bottom-right": {"x": 1.0, "y": 0.0, "xanchor": "right", "yanchor": "bottom"},
        "bottom-left": {"x": 0.0, "y": 0.0, "xanchor": "left", "yanchor": "bottom"},
        "right-outside": {"x": 1.02, "y": 1.0, "xanchor": "left", "yanchor": "top"},
    }
    return {"showlegend": True, "legend": mapping.get(pos, mapping["top-right"])}


def _apply_2d_style(fig: go.Figure, req: dict[str, Any], *, apply_legend: bool = True) -> go.Figure:
    theme = _safe_theme(req.get("theme"))
    font_size = _parse_float(req.get("font_size"), 12.0) or 12.0
    axis_title_size = _parse_float(req.get("axis_title_size"), font_size + 1.0)
    grid_on = _flag_on(req.get("grid_on"), default=True)
    axis_box_on = _flag_on(req.get("axis_box_on"), default=True)
    axis_box_width = _parse_float(req.get("axis_box_width"), 1.5)
    if axis_box_width is None:
        axis_box_width = 1.5
    axis_box_width = max(0.0, min(8.0, float(axis_box_width)))
    log_scale = str(req.get("log_scale") or "none").strip().lower()
    tick_x = _parse_float(req.get("tick_spacing_x"), None)
    tick_y = _parse_float(req.get("tick_spacing_y"), None)

    fig.update_layout(
        template=theme,
        font={"size": font_size, "color": "black"},
        title={"font": {"color": "black"}},
        margin={"l": 56, "r": 20, "t": 44, "b": 52},
    )
    xaxis_cfg: dict[str, Any] = {
        "showgrid": grid_on,
        "showline": axis_box_on,
        "linecolor": "black",
        "linewidth": axis_box_width,
        "mirror": axis_box_on,
        "ticks": "outside",
        "tickcolor": "black",
    }
    yaxis_cfg: dict[str, Any] = {
        "showgrid": grid_on,
        "showline": axis_box_on,
        "linecolor": "black",
        "linewidth": axis_box_width,
        "mirror": axis_box_on,
        "ticks": "outside",
        "tickcolor": "black",
    }
    if axis_title_size is not None:
        xaxis_cfg["title_font"] = {"size": axis_title_size}
        yaxis_cfg["title_font"] = {"size": axis_title_size}
    xaxis_cfg["title_font"] = {**(xaxis_cfg.get("title_font") or {}), "color": "black"}
    yaxis_cfg["title_font"] = {**(yaxis_cfg.get("title_font") or {}), "color": "black"}
    xaxis_cfg["tickfont"] = {"color": "black"}
    yaxis_cfg["tickfont"] = {"color": "black"}
    if tick_x is not None:
        xaxis_cfg["dtick"] = tick_x
    if tick_y is not None:
        yaxis_cfg["dtick"] = tick_y
    if log_scale in {"x", "both"}:
        xaxis_cfg["type"] = "log"
    if log_scale in {"y", "both"}:
        yaxis_cfg["type"] = "log"
    fig.update_xaxes(**xaxis_cfg)
    fig.update_yaxes(**yaxis_cfg)
    if apply_legend:
        if _flag_on(req.get("show_legend"), default=True):
            fig.update_layout(**_legend_layout(str(req.get("legend_position") or "top-right")))
        else:
            fig.update_layout(showlegend=False)
    return fig


def _apply_3d_style(fig: go.Figure, req: dict[str, Any], *, apply_legend: bool = False) -> go.Figure:
    theme = _safe_theme(req.get("theme"))
    font_size = _parse_float(req.get("font_size"), 12.0) or 12.0
    axis_title_size = _parse_float(req.get("axis_title_size"), font_size + 1.0)
    grid_on = _flag_on(req.get("grid_on"), default=True)

    scene_cfg: dict[str, Any] = {
        "xaxis": {"showgrid": grid_on},
        "yaxis": {"showgrid": grid_on},
        "zaxis": {"showgrid": grid_on},
    }
    if axis_title_size is not None:
        scene_cfg["xaxis"]["title"] = {"font": {"size": axis_title_size}}
        scene_cfg["yaxis"]["title"] = {"font": {"size": axis_title_size}}
        scene_cfg["zaxis"]["title"] = {"font": {"size": axis_title_size}}
    fig.update_layout(
        template=theme,
        font={"size": font_size},
        scene=scene_cfg,
        margin={"l": 24, "r": 18, "t": 44, "b": 24},
    )
    if apply_legend:
        if _flag_on(req.get("show_legend"), default=True):
            fig.update_layout(**_legend_layout(str(req.get("legend_position") or "top-right")))
        else:
            fig.update_layout(showlegend=False)
    return fig


def _render_pipeline_tree(snapshot: dict[str, Any], selected_node_id: str | None) -> list[Any]:
    nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
    if not isinstance(nodes, dict):
        nodes = {}
    dataset_node = _latest_dataset_node(snapshot)
    engine_text = "(not loaded)"
    if dataset_node:
        meta = dataset_node.get("metadata", {})
        dataset = meta.get("dataset", {}) if isinstance(meta, dict) else {}
        raw_engine = str(dataset.get("engine_override") or dataset.get("engine_detected") or "(not loaded)")
        engine_text = _engine_display_name(raw_engine) if raw_engine != "(not loaded)" else raw_engine

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
        prefix = "    " * depth + ("â””â”€ " if depth > 0 else "")
        cls = "rk-tree-node selected" if selected else "rk-tree-node"
        prefix = ("   " * (depth - 1) + "|_") if depth > 0 else ""
        return html.Button(
            [
                html.Span(prefix, className="rk-tree-prefix"),
                html.Span("ðŸ“", className="rk-tree-icon"),
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
        rendered.append(row(aid, _analysis_display_label(snapshot, aid), 3, str(node.get("status", "idle"))))
        meta = node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {}
        has_result = bool(node.get("result_ref") or meta.get("last_artifact_id"))
        analysis_utils = utilities_by_analysis.get(aid, [])
        analysis_presentations = visualizations_by_analysis.get(aid, [])
        if has_result or analysis_utils or analysis_presentations:
            rendered.append(row(f"virtual:utilities:{aid}", "Utilities", 4))
            for unode in analysis_utils:
                uid = str(unode.get("id"))
                rendered.append(row(uid, _utility_display_label(snapshot, uid), 5, str(unode.get("status", "idle"))))
            rendered.append(row(f"virtual:visualization:{aid}", "Presentation", 4))
            for vnode in analysis_presentations:
                label = _visualization_display_label(snapshot, str(vnode.get("id")))
                rendered.append(row(str(vnode.get("id")), label, 5, str(vnode.get("status", "idle"))))
    return rendered


def _build_canvas_payload(
    *,
    snapshot: dict[str, Any] | None,
    session: dict[str, Any] | None,
    result_store: dict[str, Any] | None,
    tab_node_id: str | None,
    x_col: str | None,
    y_col: str | None,
    group_col: str | None,
    hist_col: str | None,
    view3d_x: str | None,
    view3d_y: str | None,
    view3d_z: str | None,
    view3d_color: str | None,
    focus_atom: str | None,
) -> tuple[str, list[dict[str, Any]], go.Figure | None]:
    source_node = _resolve_canvas_source_node(snapshot, session, tab_node_id)
    if not isinstance(source_node, dict):
        return "none", [], None
    nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
    source_node_id = str(source_node.get("id") or "")
    artifact = _find_source_artifact(snapshot, source_node_id, result_store or {})
    rows = _artifact_rows(artifact if isinstance(artifact, dict) else None)
    if focus_atom:
        rows = [r for r in rows if str(r.get("atom_id")) == str(focus_atom)]
    req = source_node.get("request", {}) if isinstance(source_node.get("request"), dict) else {}
    rows = _apply_row_filters(rows, req.get("row_filters"))
    if str(source_node.get("kind")) == "utility":
        return "table", rows, None
    viz_type = str(req.get("visualization_type") or "plot2d").lower()
    if viz_type == "table":
        return "table", rows, None

    presentation_spec = None
    meta = source_node.get("metadata", {})
    if isinstance(meta, dict):
        presentation_spec = meta.get("presentation_spec")

    if viz_type == "plot2d":
        cols = infer_columns(rows)
        numeric_cols = infer_numeric_columns(rows)
        fallback_x = "iter" if "iter" in cols else ("frame_index" if "frame_index" in cols else (cols[0] if cols else ""))
        fallback_y = next((col for col in numeric_cols if col != fallback_x), numeric_cols[0] if numeric_cols else (cols[1] if len(cols) > 1 else fallback_x))
        use_x = str(req.get("x_col") or x_col or fallback_x)
        use_y = str(req.get("y_col") or y_col or fallback_y)
        use_group = str(req.get("group_col") or group_col or "")
        use_group_agg = _normalize_group_agg(req.get("group_agg"))
        use_color = str(req.get("line_color_rgb") or req.get("line_color") or "")
        try:
            use_width = float(req.get("line_width")) if req.get("line_width") is not None else None
        except Exception:
            use_width = None
        use_marker = _parse_float(req.get("marker_size"), 6.0)
        show_markers = _flag_on(req.get("show_markers"), default=False)
        trace_styles = _trace_styles_map(req.get("trace_styles"))
        plot_rows = _cached_aggregate_plot2d_rows(
            rows,
            artifact_id=str((artifact or {}).get("id") if isinstance(artifact, dict) else ""),
            x_col=use_x,
            y_col=use_y,
            group_col=use_group,
            agg=use_group_agg,
            filter_signature=json.dumps(_row_filters_from_raw(req.get("row_filters")), sort_keys=True, ensure_ascii=True),
        )
        fig = render_figure(
            plot_rows,
            presentation=presentation_spec if isinstance(presentation_spec, dict) else None,
            x_col=use_x,
            y_col=use_y,
            group_col=use_group,
            view_type="plot2d",
        )
        if fig is None:
            return "plot", rows, None
        scatter_count = sum(1 for tr in fig.data if isinstance(tr, go.Scatter))
        apply_fixed_line_color = bool(use_color) and scatter_count <= 1 and not bool(trace_styles)
        for curve_index, tr in enumerate(fig.data):
            if isinstance(tr, go.Scatter):
                trace_name = str(tr.name or f"curve_{curve_index}")
                curve_style = _trace_style_for_trace(trace_styles, trace_name, curve_index)
                curve_color = str(curve_style.get("line_color_rgb") or curve_style.get("line_color") or "").strip()
                curve_width = _parse_float(curve_style.get("line_width"), None)
                line_update: dict[str, Any] = {}
                if curve_color:
                    line_update["color"] = curve_color
                elif apply_fixed_line_color:
                    line_update["color"] = str(use_color)
                if curve_width is not None:
                    line_update["width"] = float(curve_width)
                elif use_width is not None:
                    line_update["width"] = float(use_width)
                if line_update:
                    tr.update(line=line_update)
        for curve_index, tr in enumerate(fig.data):
            if not isinstance(tr, go.Scatter):
                continue
            trace_name = str(tr.name or f"curve_{curve_index}")
            curve_style = _trace_style_for_trace(trace_styles, trace_name, curve_index)
            curve_marker_size = _parse_float(curve_style.get("marker_size"), None)
            has_curve_marker_flag = "show_markers" in curve_style
            curve_show_markers = _flag_on(curve_style.get("show_markers"), default=False) if has_curve_marker_flag else show_markers
            mode = str(tr.mode or "lines")
            marker_size = curve_marker_size if curve_marker_size is not None else use_marker
            if curve_show_markers and marker_size is not None and marker_size > 0:
                if "markers" not in mode:
                    mode = f"{mode}+markers" if mode else "markers"
                tr.update(mode=mode, marker={"size": float(marker_size)})
            else:
                mode_clean = mode.replace("+markers", "").replace("markers+", "")
                tr.update(mode=(mode_clean if mode_clean else "lines"))
        _apply_2d_style(fig, req, apply_legend=True)
        return "plot", rows, fig

    if viz_type == "histogram":
        cols = infer_columns(rows)
        numeric_cols = infer_numeric_columns(rows)
        fallback_hist = numeric_cols[0] if numeric_cols else (cols[0] if cols else "")
        use_hist = str(req.get("x_col") or req.get("y_col") or hist_col or fallback_hist)
        fig = render_figure(
            rows,
            presentation=presentation_spec if isinstance(presentation_spec, dict) else None,
            x_col=use_hist,
            y_col=use_hist,
            view_type="histogram",
        )
        if fig is None:
            return "plot", rows, None
        hist_color = str(req.get("line_color_rgb") or req.get("line_color") or "")
        if hist_color:
            for tr in fig.data:
                if isinstance(tr, go.Histogram):
                    tr.update(marker={"color": hist_color})
        _apply_2d_style(fig, req, apply_legend=True)
        return "plot", rows, fig

    fig3d = render_figure(
        rows,
        presentation=presentation_spec if isinstance(presentation_spec, dict) else None,
        x_col=str(req.get("x_col") or view3d_x or ""),
        y_col=str(req.get("y_col") or view3d_y or ""),
        z_col=str(req.get("z_col") or view3d_z or ""),
        color_col=str(req.get("color_col") or view3d_color or ""),
        view_type="scatter3d",
    )
    if fig3d is None:
        return "plot", rows, None
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
    return "plot", rows, fig3d


