"""Dash callbacks for Phase 4 workflow (pipeline, utilities, and 3D views)."""

from __future__ import annotations

import logging
from typing import Any
from numbers import Number
from datetime import date, datetime
import statistics

from dash import ALL, Input, Output, State, ctx, dash_table, dcc, html, no_update
import plotly.graph_objects as go
import plotly.io as pio
import os
import json
from pathlib import Path
import tempfile

from reaxkit.core.analysis_cli_routing_registry import get_registered_analysis_commands
from reaxkit.core.analysis_task_registry import TASK_LABELS, TASK_REGISTRY, task_display_label
from reaxkit.presentation.specs import ensure_presentation_spec, spec_to_dash_request
from reaxkit.webui.backend.api import WebUIApiService
from reaxkit.webui.backend.tabular_payload import extract_tabular_rows, infer_columns, infer_numeric_columns
from reaxkit.webui.backend.utility_registry import canonical_utility_name, default_utility_request
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


def _trace(message: str) -> None:
    text = str(message)
    try:
        print(text, flush=True)
    except Exception:
        pass
    paths: list[Path] = []
    env_path = str(os.environ.get("REAXKIT_UI_TRACE_PATH", "")).strip()
    if env_path:
        paths.append(Path(env_path))
    paths.append(Path(os.getcwd()) / "ui_trace.log")
    paths.append(Path(tempfile.gettempdir()) / "reaxkit_ui_trace.log")
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
    current_node = nodes.get(str(selected_node_id)) if isinstance(nodes, dict) else None

    # 1) direct cache
    direct = cache.get(selected_node_id)
    if isinstance(direct, dict) and isinstance(current_node, dict):
        direct_id = str(direct.get("id") or "")
        result_ref = str(current_node.get("result_ref") or "")
        meta = current_node.get("metadata", {})
        last_id = str(meta.get("last_artifact_id") or "") if isinstance(meta, dict) else ""
        # Avoid stale node->artifact cache entries after upstream transformations.
        if (result_ref and direct_id == result_ref) or (last_id and direct_id == last_id):
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
            art = cache.get(cid)
            if isinstance(art, dict):
                art_id = str(art.get("id") or "")
                if art_id == str(result_ref):
                    return art
            if isinstance(artifacts, dict):
                snap_art = artifacts.get(str(result_ref))
                if isinstance(snap_art, dict):
                    return snap_art
        meta = current.get("metadata", {})
        if isinstance(meta, dict):
            last_id = meta.get("last_artifact_id")
            if last_id:
                art = cache.get(cid)
                if isinstance(art, dict):
                    art_id = str(art.get("id") or "")
                    if art_id == str(last_id):
                        return art
                if isinstance(artifacts, dict):
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
    log_scale = str(req.get("log_scale") or "none").strip().lower()
    tick_x = _parse_float(req.get("tick_spacing_x"), None)
    tick_y = _parse_float(req.get("tick_spacing_y"), None)

    fig.update_layout(template=theme, font={"size": font_size})
    xaxis_cfg: dict[str, Any] = {"showgrid": grid_on}
    yaxis_cfg: dict[str, Any] = {"showgrid": grid_on}
    if axis_title_size is not None:
        xaxis_cfg["title_font"] = {"size": axis_title_size}
        yaxis_cfg["title_font"] = {"size": axis_title_size}
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
    fig.update_layout(template=theme, font={"size": font_size}, scene=scene_cfg)
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


def register_analysis_callbacks(app, service: WebUIApiService) -> None:
    """Register analysis-focused Dash callbacks."""

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
            else str(workspace_dir or "reaxkit_workspace/")
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
            else str(cfg.get("workspace_dir") or "reaxkit_workspace/")
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
        Input("btn-add-filter-node", "n_clicks"),
        Input("btn-add-ema-node", "n_clicks"),
        Input("btn-add-sma-node", "n_clicks"),
        Input("btn-add-frame-range-node", "n_clicks"),
        Input("btn-add-transform-node", "n_clicks"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def on_add_utility_node(
        n_join: int,
        n_filter: int,
        n_ema: int,
        n_sma: int,
        n_frame_range: int,
        n_transform: int,
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
    ):
        if not session or not snapshot:
            return no_update, no_update, no_update
        trig = ctx.triggered_id
        clicks_by_trigger = {
            "btn-add-join-node": int(n_join or 0),
            "btn-add-filter-node": int(n_filter or 0),
            "btn-add-ema-node": int(n_ema or 0),
            "btn-add-sma-node": int(n_sma or 0),
            "btn-add-frame-range-node": int(n_frame_range or 0),
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
            "btn-add-filter-node": "filter_rows",
            "btn-add-ema-node": "denoise_ema",
            "btn-add-sma-node": "denoise_sma",
            "btn-add-frame-range-node": "frame_range",
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
            service.delete_node(pipeline_id, node_id)
        except Exception as exc:
            return no_update, no_update, no_update, f"ERROR: Delete failed: {exc}"

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
        return service.get_pipeline(pipeline_id)

    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Output("result-store", "data", allow_duplicate=True),
        Output("execute-loading-proxy", "children", allow_duplicate=True),
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
        _trace(f"[UI_TRACE] on_apply_node called n_clicks={n_clicks} has_session={bool(session)} has_snapshot={bool(snapshot)}")
        if not n_clicks or not session or not snapshot:
            return no_update, no_update, no_update, no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        node = _selected_node(snapshot, session)
        if not pipeline_id or not node:
            return no_update, no_update, no_update, "WARN: Select a node first"

        node_id = str(node.get("id") or "")
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
            next_store[node_id] = artifact
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
                    if analysis_id:
                        next_store[str(analysis_id)] = artifact
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
        return next_snapshot, next_store, html.Span(str(n_clicks), style={"display": "none"}), "Node executed"

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

        selected_node = nodes.get(selected_id)
        viz_nodes: list[dict[str, Any]] = []
        if selected_id.startswith("virtual:visualization:"):
            analysis_id = selected_id.split(":", 2)[2]
            viz_nodes = _visualization_nodes_for_analysis(snapshot, analysis_id)
        else:
            if isinstance(selected_node, dict) and str(selected_node.get("kind")) == "visualization":
                viz_nodes = [selected_node]
            elif isinstance(selected_node, dict) and str(selected_node.get("kind")) == "utility":
                analysis_id = _ancestor_analysis_id(nodes, selected_id)
                viz_nodes = _visualization_nodes_for_analysis(snapshot, analysis_id) if analysis_id else []
            elif isinstance(selected_node, dict) and str(selected_node.get("kind")) == "analysis":
                viz_nodes = []

        if not viz_nodes:
            return [], None
        tabs = [
            dcc.Tab(
                label=_visualization_display_label(snapshot, str(v.get("id"))),
                value=str(v.get("id")),
            )
            for v in viz_nodes
        ]
        valid = {str(v.get("id")) for v in viz_nodes}
        if selected_id in valid:
            return tabs, selected_id
        if isinstance(selected_node, dict) and str(selected_node.get("kind")) == "utility":
            table_tab_id = ""
            for vnode in viz_nodes:
                req = vnode.get("request", {}) if isinstance(vnode.get("request"), dict) else {}
                meta = vnode.get("metadata", {}) if isinstance(vnode.get("metadata"), dict) else {}
                vtype = _canonical_viz_type(req.get("visualization_type") or meta.get("visualization_type") or vnode.get("name"))
                if vtype == "table":
                    table_tab_id = str(vnode.get("id"))
                    break
            if table_tab_id:
                return tabs, table_tab_id
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
            filter_label = str(utility_specs.get("filter_rows", {}).get("label") or "Filter rows")
            ema_label = str(utility_specs.get("denoise_ema", {}).get("label") or "Denoise (EMA)")
            sma_label = str(utility_specs.get("denoise_sma", {}).get("label") or "Denoise (SMA)")
            frame_label = str(utility_specs.get("frame_range", {}).get("label") or "Frame range")
            transform_label = str(utility_specs.get("column_transform", {}).get("label") or "Column transform")
            return html.Div(
                [
                    html.Div("Node: Utilities"),
                    html.Button(join_label, id="btn-add-join-node", n_clicks=0),
                    html.Button(filter_label, id="btn-add-filter-node", n_clicks=0),
                    html.Button(ema_label, id="btn-add-ema-node", n_clicks=0),
                    html.Button(sma_label, id="btn-add-sma-node", n_clicks=0),
                    html.Button(frame_label, id="btn-add-frame-range-node", n_clicks=0),
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
                        dcc.Dropdown(id="viz-group-col", options=opts, value=None, clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-group-agg", options=[{"label": "none", "value": "none"}], value="none", clearable=False, style={"display": "none"}),
                    ]
                )
            elif draft_type == "histogram":
                body.extend(
                    [
                        html.Label("value content"),
                        dcc.Dropdown(id="viz-x-col", options=opts, value=hist_default, clearable=False),
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
            field_lines: list[Any] = []
            for field in utility_spec.get("fields", []) if isinstance(utility_spec, dict) else []:
                if not isinstance(field, dict):
                    continue
                fname = str(field.get("name") or "").strip()
                if not fname:
                    continue
                ftype = str(field.get("type") or "Any")
                fdefault = field.get("default")
                field_lines.append(
                    html.Div(
                        [
                            html.Code(fname),
                            html.Span(f" : {ftype}"),
                            html.Span(f"  default={fdefault!r}" if fdefault is not None else "  default=None"),
                        ]
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
            lines.extend(
                [
                    html.Div("Edit request JSON", className="rk-subtitle"),
                    dcc.Textarea(
                        id="utility-request-json",
                        value=json.dumps(merged_req, indent=2, ensure_ascii=True),
                        style={"width": "100%", "minHeight": "180px", "fontFamily": "Consolas, Courier New, monospace"},
                    ),
                    html.Div("Schema", className="rk-subtitle"),
                    html.Div(field_lines or [html.Div("No schema metadata found for this utility.")], className="rk-stack"),
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
                        html.Div("Table filtering/sorting is available directly in Result Tabs.", className="rk-subtitle"),
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
            "line_color": str(line_color_name or "blue"),
            "line_color_rgb": str(line_color_rgb or ""),
            "line_width": float(line_width if line_width is not None else 2.0),
            "font_size": float(font_size if font_size is not None else 12.0),
            "marker_size": float(marker_size if marker_size is not None else 0.0),
            "theme": _safe_theme(theme or "plotly_white"),
            "axis_title_size": float(axis_title_size if axis_title_size is not None else 13.0),
            "grid_on": bool("on" in (grid_on_values or [])),
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
                "line_color": str(line_color_name or payload.get("line_color") or "blue"),
                "line_color_rgb": str(line_color_rgb or payload.get("line_color_rgb") or ""),
                "line_width": float(line_width if line_width is not None else float(payload.get("line_width") or 2.0)),
                "font_size": float(font_size if font_size is not None else float(payload.get("font_size") or 12.0)),
                "marker_size": float(marker_size if marker_size is not None else float(payload.get("marker_size") or 0.0)),
                "theme": _safe_theme(theme or payload.get("theme") or "plotly_white"),
                "axis_title_size": float(axis_title_size if axis_title_size is not None else float(payload.get("axis_title_size") or 13.0)),
                "grid_on": bool("on" in (grid_on_values or [])),
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
        service.update_node(
            pipeline_id,
            str(node["id"]),
            {"request": payload, "metadata": {"visualization_type": payload["visualization_type"]}},
        )
        return service.get_pipeline(pipeline_id)

    @app.callback(
        Output("selected-curve-store", "data"),
        Input("session-store", "data"),
        Input({"type": "plot-graph", "slot": ALL}, "clickData"),
        State({"type": "plot-graph", "slot": ALL}, "id"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def on_curve_click(
        session: dict[str, Any] | None,
        click_data_all: list[dict[str, Any] | None] | None,
        graph_ids: list[dict[str, Any]] | None,
        snapshot: dict[str, Any] | None,
    ):
        triggered = ctx.triggered_id
        if triggered == "session-store":
            return {}
        if not session or not isinstance(click_data_all, list):
            return no_update
        node = _selected_node(snapshot, session)
        req = node.get("request", {}) if isinstance(node, dict) and isinstance(node.get("request"), dict) else {}
        if not isinstance(node, dict) or str(node.get("kind")) != "visualization":
            return no_update
        if str(req.get("visualization_type") or "plot2d").lower() != "plot2d":
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
        return service.get_pipeline(pipeline_id)

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
        service.update_node(
            pipeline_id,
            str(node["id"]),
            {"request": payload, "metadata": {"visualization_type": payload["visualization_type"]}},
        )
        return service.get_pipeline(pipeline_id)

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
        return str(current_value or "reaxkit_workspace/"), False

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
            empty = "No presentation selected. Select a presentation node under this analysis."
            return empty, empty
        if str(node_id).startswith("virtual:visualization:") and not tab_node_id:
            empty = "No presentations yet for this analysis."
            return empty, empty

        source_node_id = str(tab_node_id or node_id)
        source_node = nodes.get(source_node_id) if isinstance(nodes, dict) else None
        if not isinstance(source_node, dict):
            empty = "No presentation selected."
            return empty, empty
        if str(source_node.get("kind")) == "utility":
            analysis_id = _ancestor_analysis_id(nodes, source_node_id) if isinstance(nodes, dict) else None
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
                    source_node_id = str(source_node.get("id"))
        if str(source_node.get("kind")) == "utility":
            artifact = _find_source_artifact(snapshot, source_node_id, result_store or {})
            rows = _artifact_rows(artifact if isinstance(artifact, dict) else None)
            if not rows:
                content = "No result rows yet."
                return content, content
            return _build_result_table(rows, max_rows=200), _build_result_table(rows, max_rows=200)
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
            table_max_rows = int(req.get("table_max_rows") or 200)
            return _build_result_table(rows, max_rows=table_max_rows), _build_result_table(rows, max_rows=table_max_rows)

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
            plot_rows = _aggregate_plot2d_rows(
                rows,
                x_col=use_x,
                y_col=use_y,
                group_col=use_group,
                agg=use_group_agg,
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
                content = "No plottable data."
                return content, content
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
                curve_show_markers = (
                    _flag_on(curve_style.get("show_markers"), default=False)
                    if has_curve_marker_flag
                    else show_markers
                )
                mode = str(tr.mode or "lines")
                marker_size = curve_marker_size if curve_marker_size is not None else use_marker
                if curve_show_markers and marker_size is not None and marker_size > 0:
                    if "markers" not in mode:
                        mode = f"{mode}+markers" if mode else "markers"
                    tr.update(mode=mode, marker={"size": float(marker_size)})
                else:
                    mode_clean = mode.replace("+markers", "").replace("markers+", "")
                    mode_clean = mode_clean if mode_clean else "lines"
                    tr.update(mode=mode_clean)
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
            graph_result = dcc.Graph(id={"type": "plot-graph", "slot": "result"}, figure=fig, config={"displaylogo": False})
            graph_canvas = dcc.Graph(
                id={"type": "plot-graph", "slot": "canvas"},
                figure=go.Figure(fig),
                config={"displaylogo": False},
            )
            return graph_result, graph_canvas

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
                content = "No numeric data for histogram."
                return content, content
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
            graph_result = dcc.Graph(id={"type": "plot-graph", "slot": "result"}, figure=fig, config={"displaylogo": False})
            graph_canvas = dcc.Graph(
                id={"type": "plot-graph", "slot": "canvas"},
                figure=go.Figure(fig),
                config={"displaylogo": False},
            )
            return graph_result, graph_canvas

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
            content = "No 3D data."
            return content, content
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
        graph3d_result = dcc.Graph(id={"type": "plot-graph", "slot": "result"}, figure=fig3d, config={"displaylogo": False})
        graph3d_canvas = dcc.Graph(
            id={"type": "plot-graph", "slot": "canvas"},
            figure=go.Figure(fig3d),
            config={"displaylogo": False},
        )
        return graph3d_result, graph3d_canvas
