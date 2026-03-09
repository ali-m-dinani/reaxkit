"""Dash callbacks for Phase 4 workflow (pipeline, utilities, and 3D views)."""

from __future__ import annotations

from typing import Any
from numbers import Number
from datetime import date, datetime

from dash import ALL, Input, Output, State, ctx, dash_table, dcc, html, no_update
import plotly.graph_objects as go
import plotly.io as pio
import os
import subprocess
from pathlib import Path
import re
import json
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from importlib import metadata as importlib_metadata

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


def _build_result_table(rows: list[dict[str, Any]], *, max_rows: int = 200) -> dash_table.DataTable:
    safe_max = max(10, min(10000, int(max_rows)))
    column_ids = [str(c) for c in (rows[0].keys() if rows else [])]
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


def _default_workspace_dir_for_dataset(dataset_path: str | None) -> str:
    base = Path(str(dataset_path or ".").strip() or ".")
    if not base.is_absolute():
        base = (Path.cwd() / base).resolve()
    return str((base / "reaxkit_workspace").resolve())


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


def _resolve_logs_root(config: dict[str, Any] | None) -> Path:
    cfg = dict(config or {})
    workspace_dir = str(cfg.get("workspace_dir") or "").strip()
    candidates: list[Path] = []
    if workspace_dir:
        p = Path(workspace_dir)
        candidates.append(p)
        if not p.is_absolute():
            candidates.append((Path.cwd() / p).resolve())
            candidates.append((Path(__file__).resolve().parent / p).resolve())
    candidates.extend(
        [
            Path("reaxkit_workkspace"),
            Path("reaxkit_workspace"),
            Path.cwd() / "reaxkit_workkspace",
            Path.cwd() / "reaxkit_workspace",
            Path(__file__).resolve().parent / "reaxkit_workkspace",
            Path(__file__).resolve().parent / "reaxkit_workspace",
        ]
    )
    seen: set[str] = set()
    deduped: list[Path] = []
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    for root in deduped:
        logs = root / "logs"
        if logs.exists() and logs.is_dir():
            return logs
    return (deduped[0] / "logs") if deduped else (Path.cwd() / "reaxkit_workkspace" / "logs")


def _pick_latest(paths: list[Path]) -> Path | None:
    existing = [p for p in paths if p.exists() and p.is_file()]
    if not existing:
        return None
    existing.sort(key=lambda p: p.stat().st_mtime)
    return existing[-1]


def _read_tail(path: Path | None, *, max_lines: int = 400) -> str:
    if path is None or not path.exists():
        return "No log file found."
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Failed to read log file: {exc}"
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return "\n".join(lines) if lines else "(empty log)"


def _select_log_files(config: dict[str, Any] | None) -> tuple[Path | None, Path | None]:
    logs_root = _resolve_logs_root(config)
    human_candidates = sorted(logs_root.glob("run_*.timing.log"))
    human = _pick_latest(human_candidates) or _pick_latest([logs_root / "timing_human.log"])

    low = _pick_latest([logs_root / "timing.log"])
    if low is not None:
        return human, low
    run_low_candidates = [p for p in logs_root.glob("run_*.log") if not p.name.endswith(".timing.log")]
    run_low_candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
    low = _pick_latest(run_low_candidates)
    if low is None:
        low = _pick_latest([logs_root / "reaxkit.log"])
    return human, low


def _default_repo_slug() -> str:
    # Preferred explicit repo; fallback to local git origin if available.
    explicit = os.environ.get("REAXKIT_GITHUB_REPO", "").strip()
    if explicit:
        return explicit
    try:
        cfg = Path(__file__).resolve().parents[3] / ".git" / "config"
        if cfg.exists():
            text = cfg.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"url\s*=\s*https?://github\.com/([^/\s]+/[^/\s]+?)(?:\.git)?\s*$", text, re.MULTILINE)
            if match:
                return str(match.group(1))
    except Exception:
        pass
    return "ali-m-dinani/reaxkit"


def _installed_reaxkit_version() -> str:
    try:
        return importlib_metadata.version("reaxkit")
    except Exception:
        try:
            pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
            if pyproject.exists():
                text = pyproject.read_text(encoding="utf-8", errors="replace")
                m = re.search(r'^\s*version\s*=\s*"([^"]+)"\s*$', text, re.MULTILINE)
                if m:
                    return str(m.group(1))
        except Exception:
            pass
    return "unknown"


def _version_tuple(raw: str) -> tuple[int, ...]:
    txt = str(raw or "").strip().lstrip("vV")
    parts = re.split(r"[^0-9]+", txt)
    nums = [int(p) for p in parts if p.isdigit()]
    return tuple(nums) if nums else (0,)


def _fetch_latest_release_tag(repo_slug: str) -> tuple[str, str]:
    def _latest_tag_from_git() -> str | None:
        commands = [
            # Prefer configured local remote (works with private repos when git auth is set up).
            ["git", "-C", str(Path(__file__).resolve().parents[3]), "ls-remote", "--tags", "origin"],
            # Fallback to direct GitHub URL.
            ["git", "ls-remote", "--tags", f"https://github.com/{repo_slug}.git"],
        ]
        tags: set[str] = set()
        for cmd in commands:
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
            except Exception:
                continue
            if proc.returncode != 0 or not proc.stdout:
                continue
            for raw in proc.stdout.splitlines():
                parts = raw.strip().split()
                if len(parts) < 2:
                    continue
                ref = str(parts[1])
                if not ref.startswith("refs/tags/"):
                    continue
                tag = ref.split("refs/tags/", 1)[1]
                if tag.endswith("^{}"):
                    tag = tag[:-3]
                if tag:
                    tags.add(tag)
            if tags:
                break
        if not tags:
            return None
        return sorted(tags, key=lambda t: (_version_tuple(t), t))[-1]

    api_url = f"https://api.github.com/repos/{repo_slug}/releases/latest"
    req = Request(api_url, headers={"User-Agent": "ReaxKit-WebUI"})
    try:
        with urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        tag = str(payload.get("tag_name") or payload.get("name") or "").strip()
        if tag:
            return tag, "release"
    except HTTPError as exc:
        if int(getattr(exc, "code", 0)) != 404:
            raise

    # Fallback for repos without Releases endpoint support.
    try:
        tag_req = Request(f"https://api.github.com/repos/{repo_slug}/tags?per_page=1", headers={"User-Agent": "ReaxKit-WebUI"})
        with urlopen(tag_req, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        if isinstance(payload, list) and payload:
            tag = str((payload[0] or {}).get("name") or "").strip()
            if tag:
                return tag, "tag"
    except HTTPError as exc:
        if int(getattr(exc, "code", 0)) != 404:
            raise

    git_tag = _latest_tag_from_git()
    if git_tag:
        return git_tag, "tag"
    raise ValueError("No release/tag found (GitHub API and git tag fallback failed).")


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
    raw_name = str(node.get("metadata", {}).get("task_name") or node.get("name") or "analysis")
    key = raw_name.strip().lower()
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


def _viz_request_with_defaults(existing: dict[str, Any] | None, viz_type: str) -> dict[str, Any]:
    req = dict(existing or {})
    req["visualization_type"] = str(viz_type)
    req.setdefault("x_col", "iter")
    req.setdefault("y_col", "msd")
    req.setdefault("z_col", "frame_index")
    req.setdefault("use_plot_title", False)
    req.setdefault("plot_title", "")
    req.setdefault("x_title", "")
    req.setdefault("y_title", "")
    req.setdefault("z_title", "")
    req.setdefault("color_col", "")
    req.setdefault("group_col", "atom_id")
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
            {
                "dataset_path": os.getcwd(),
                "engine_name": "autodetect",
                "role_xmolout": "xmolout",
                "workspace_default": True,
                "workspace_dir": _default_workspace_dir_for_dataset(os.getcwd()),
                "draft_viz_type": "plot2d",
            },
            "Ready",
        )

    @app.callback(
        Output("ui-store", "data"),
        Input("btn-nav-analysis", "n_clicks"),
        Input("btn-nav-log", "n_clicks"),
        State("ui-store", "data"),
        prevent_initial_call=True,
    )
    def on_nav_page_change(
        _n_analysis: int,
        _n_log: int,
        ui_data: dict[str, Any] | None,
    ):
        trig = str(ctx.triggered_id or "")
        current = str((ui_data or {}).get("page") or "analysis")
        if trig == "btn-nav-log":
            return {"page": "log"}
        if trig == "btn-nav-analysis":
            return {"page": "analysis"}
        return {"page": current}

    @app.callback(
        Output("panel-left", "style"),
        Output("panel-canvas", "style"),
        Output("panel-props", "style"),
        Output("panel-results", "style"),
        Output("panel-info", "style"),
        Output("panel-log-page", "style"),
        Output("btn-nav-analysis", "className"),
        Output("btn-nav-log", "className"),
        Input("ui-store", "data"),
        prevent_initial_call=False,
    )
    def render_active_page(ui_data: dict[str, Any] | None):
        page = str((ui_data or {}).get("page") or "analysis").lower()
        show_analysis = page == "analysis"
        show_log = page == "log"
        base = "rk-nav-btn"
        return (
            {} if show_analysis else {"display": "none"},
            {} if show_analysis else {"display": "none"},
            {} if show_analysis else {"display": "none"},
            {} if show_analysis else {"display": "none"},
            {} if show_analysis else {"display": "none"},
            {"display": "block"} if show_log else {"display": "none"},
            f"{base} active" if show_analysis else base,
            f"{base} active" if show_log else base,
        )

    @app.callback(
        Output("help-update-status", "children"),
        Input("btn-help-check-updates", "n_clicks"),
        prevent_initial_call=True,
    )
    def on_help_check_updates(n_clicks: int):
        if not n_clicks:
            return no_update
        repo = _default_repo_slug()
        local = _installed_reaxkit_version()
        try:
            remote, source = _fetch_latest_release_tag(repo)
        except (URLError, HTTPError, ValueError) as exc:
            return f"Update check failed: {exc}"
        except Exception as exc:
            return f"Update check failed: {exc}"

        local_t = _version_tuple(local)
        remote_t = _version_tuple(remote)
        source_label = "release" if source == "release" else "tag"
        if local == "unknown":
            return f"Installed: unknown | Latest {source_label}: {remote} ({repo})"
        if local_t < remote_t:
            return f"Update available: installed {local} -> latest {source_label} {remote} ({repo})"
        if local_t > remote_t:
            return f"Installed {local} is newer than latest {source_label} {remote} ({repo})"
        return f"Up to date: {local} ({source_label} {remote}, {repo})"

    @app.callback(
        Output("log-human-name", "children"),
        Output("log-human-content", "children"),
        Output("log-low-name", "children"),
        Output("log-low-content", "children"),
        Input("ui-store", "data"),
        Input("pipeline-store", "data"),
        State("config-store", "data"),
        prevent_initial_call=False,
    )
    def render_log_page(
        ui_data: dict[str, Any] | None,
        _snapshot: dict[str, Any] | None,
        config_data: dict[str, Any] | None,
    ):
        page = str((ui_data or {}).get("page") or "analysis").lower()
        if page != "log":
            return no_update, no_update, no_update, no_update
        human, low = _select_log_files(config_data)
        human_name = human.name if human else "(missing)"
        low_name = low.name if low else "(missing)"
        human_content = _read_tail(human)
        low_content = _read_tail(low)
        return human_name, human_content, low_name, low_content

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
        dataset_node = _latest_dataset_node(snapshot)
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
        trig = ctx.triggered_id
        if trig == "btn-add-filter-node" and int(n_filter or 0) <= 0:
            return no_update, no_update, no_update
        if trig == "btn-add-ema-node" and int(n_ema or 0) <= 0:
            return no_update, no_update, no_update
        if trig == "btn-add-sma-node" and int(n_sma or 0) <= 0:
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
                "parent_id": parent_id or node["id"],
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
                    "plot_title": "",
                    "x_title": "",
                    "y_title": "",
                    "z_title": "",
                    "color_col": str(color_col or ""),
                    "group_col": str(group_col or ""),
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
        Input("msd-atom-ids", "value"),
        Input("msd-atom-types", "value"),
        Input("msd-dims", "value"),
        Input("msd-origin", "value"),
        Input("msd-frames", "value"),
        Input("msd-every", "value"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def on_live_msd_params(
        atom_ids_raw: str | None,
        atom_types_raw: str | None,
        dims_raw: str | None,
        origin_raw: str | None,
        frames_raw: str | None,
        every_raw: str | None,
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
    ):
        if not session or not snapshot:
            return no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        node = _selected_node(snapshot, session)
        if not pipeline_id or not node or str(node.get("kind")) != "analysis" or str(node.get("name", "")).lower() != "msd":
            return no_update

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
        old_req = node.get("request", {}) if isinstance(node.get("request"), dict) else {}
        if old_req == request_payload:
            return no_update
        service.update_node(pipeline_id, str(node["id"]), {"request": request_payload})
        return service.get_pipeline(pipeline_id)

    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Input("util-filter-column", "value"),
        Input("util-filter-values", "value"),
        Input("util-denoise-column", "value"),
        Input("util-denoise-alpha", "value"),
        Input("util-denoise-window", "value"),
        Input("util-denoise-group", "value"),
        Input("util-denoise-xcol", "value"),
        State("session-store", "data"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def on_live_utility_params(
        filter_column: str | None,
        filter_values: str | None,
        denoise_column: str | None,
        denoise_alpha: float | None,
        denoise_window: int | None,
        denoise_group: str | None,
        denoise_xcol: str | None,
        session: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
    ):
        if not session or not snapshot:
            return no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        node = _selected_node(snapshot, session)
        if not pipeline_id or not node or str(node.get("kind")) != "utility":
            return no_update

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
            return no_update

        old_req = node.get("request", {}) if isinstance(node.get("request"), dict) else {}
        if old_req == request_payload:
            return no_update
        service.update_node(pipeline_id, str(node["id"]), {"request": request_payload})
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
        if not n_clicks or not session or not snapshot:
            return no_update, no_update, no_update, no_update
        pipeline_id = str(session.get("pipeline_id", ""))
        node = _selected_node(snapshot, session)
        if not pipeline_id or not node:
            return no_update, no_update, no_update, "WARN: Select a node first"

        node_id = str(node.get("id") or "")

        try:
            run_result = service.apply_node(pipeline_id, node_id)
        except Exception as exc:
            return no_update, no_update, html.Span(str(n_clicks), style={"display": "none"}), f"ERROR: Execute failed: {exc}"

        artifact = run_result.get("artifact") if isinstance(run_result, dict) else None
        next_store = dict(result_store or {})
        if isinstance(artifact, dict) and "id" in artifact:
            next_store[node_id] = artifact
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
    )
    def render_properties(
        snapshot: dict[str, Any] | None,
        session: dict[str, Any] | None,
        result_store_in: dict[str, Any] | None,
        config_in: dict[str, Any] | None,
    ):
        selected_id = str((session or {}).get("selected_node_id") or "")
        result_store = dict(result_store_in or {})
        config = dict(config_in or {})

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
            draft_type = str(config.get("draft_viz_type") or "plot2d")
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
                        dcc.Dropdown(id="viz-x-col", options=opts, value="iter" if "iter" in cols else cols[0], clearable=False),
                        html.Label("y axis content"),
                        dcc.Dropdown(id="viz-y-col", options=opts, value="msd" if "msd" in cols else cols[0], clearable=False),
                        html.Label("group content"),
                        dcc.Dropdown(id="viz-group-col", options=opts, value="atom_id" if "atom_id" in cols else None, clearable=True),
                        dcc.Dropdown(id="viz-z-col", options=opts, value=cols[0], clearable=False, style={"display": "none"}),
                        dcc.Dropdown(id="viz-color-col", options=opts, value=None, clearable=True, style={"display": "none"}),
                    ]
                )
            elif draft_type == "scatter3d":
                body.extend(
                    [
                        html.Label("x axis content"),
                        dcc.Dropdown(id="viz-x-col", options=opts, value=cols[0], clearable=False),
                        html.Label("y axis content"),
                        dcc.Dropdown(id="viz-y-col", options=opts, value=cols[0], clearable=False),
                        html.Label("z axis content"),
                        dcc.Dropdown(id="viz-z-col", options=opts, value=cols[0], clearable=False),
                        html.Label("color by"),
                        dcc.Dropdown(id="viz-color-col", options=opts, value=None, clearable=True),
                        dcc.Dropdown(id="viz-group-col", options=opts, value=None, clearable=True, style={"display": "none"}),
                    ]
                )
            elif draft_type == "histogram":
                body.extend(
                    [
                        html.Label("value content"),
                        dcc.Dropdown(id="viz-x-col", options=opts, value="msd" if "msd" in cols else cols[0], clearable=False),
                        dcc.Dropdown(id="viz-y-col", options=opts, value="", clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-z-col", options=opts, value="", clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-color-col", options=opts, value=None, clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-group-col", options=opts, value=None, clearable=True, style={"display": "none"}),
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

        if node.get("kind") == "analysis" and str(node.get("name", "")).lower() == "msd":
            req = node.get("request", {}) if isinstance(node.get("request", {}), dict) else {}
            is_running = str(node.get("status", "")).lower() == "running"
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
                    dcc.Input(id="util-filter-values", type="text", value="", style={"display": "none"}),
                    dcc.Dropdown(id="util-filter-column", options=[], value=None, style={"display": "none"}),
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
            util_name = str(node.get("metadata", {}).get("utility_name") or node.get("name") or "").lower()
            is_running = str(node.get("status", "")).lower() == "running"
            source_art = _find_source_artifact(snapshot, str(node.get("id")), result_store)
            source_rows = _artifact_rows(source_art if isinstance(source_art, dict) else None)
            cols = list(source_rows[0].keys()) if source_rows else ["iter", "frame_index", "msd", "atom_id"]
            col_opts = [{"label": c, "value": c} for c in cols]
            numeric_cols = [c for c in cols if any(isinstance(r.get(c), (int, float)) for r in (source_rows[:20] if source_rows else []))]
            numeric_opts = [{"label": c, "value": c} for c in (numeric_cols or cols)]
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
            if util_name in {"filter_rows", "filter_atoms"}:
                lines.extend(
                    [
                        html.Label("column"),
                        dcc.Dropdown(
                            id="util-filter-column",
                            options=col_opts,
                            value=str(req.get("column") or ("atom_id" if "atom_id" in cols else (cols[0] if cols else ""))),
                            clearable=False,
                        ),
                        html.Label("values (comma-separated)"),
                        dcc.Input(id="util-filter-values", type="text", value=str(req.get("values", ""))),
                        dcc.Dropdown(id="util-denoise-column", options=numeric_opts, value=(numeric_cols[0] if numeric_cols else (cols[0] if cols else "")), style={"display": "none"}),
                        dcc.Input(id="util-denoise-alpha", type="number", value=0.3, style={"display": "none"}),
                        dcc.Input(id="util-denoise-window", type="number", value=5, style={"display": "none"}),
                        dcc.Dropdown(id="util-denoise-group", options=col_opts, value=("atom_id" if "atom_id" in cols else (cols[0] if cols else "")), style={"display": "none"}),
                        dcc.Dropdown(id="util-denoise-xcol", options=col_opts, value=("iter" if "iter" in cols else (cols[0] if cols else "")), style={"display": "none"}),
                    ]
                )
            elif util_name == "denoise_ema":
                lines.extend(
                    [
                        html.Label("column"),
                        dcc.Dropdown(
                            id="util-denoise-column",
                            options=numeric_opts,
                            value=str(req.get("column") or ("msd" if "msd" in (numeric_cols or cols) else ((numeric_cols or cols)[0] if (numeric_cols or cols) else ""))),
                            clearable=False,
                        ),
                        html.Label("alpha"),
                        dcc.Input(id="util-denoise-alpha", type="number", value=float(req.get("alpha", 0.3)), min=0.01, max=1.0, step=0.01),
                        html.Label("group_by"),
                        dcc.Dropdown(
                            id="util-denoise-group",
                            options=col_opts,
                            value=str(req.get("group_by") or ("atom_id" if "atom_id" in cols else (cols[0] if cols else ""))),
                            clearable=False,
                        ),
                        html.Label("x_col"),
                        dcc.Dropdown(
                            id="util-denoise-xcol",
                            options=col_opts,
                            value=str(req.get("x_col") or ("iter" if "iter" in cols else (cols[0] if cols else ""))),
                            clearable=False,
                        ),
                        dcc.Input(id="util-denoise-window", type="number", value=5, style={"display": "none"}),
                    ]
                )
            elif util_name == "denoise_sma":
                lines.extend(
                    [
                        html.Label("column"),
                        dcc.Dropdown(
                            id="util-denoise-column",
                            options=numeric_opts,
                            value=str(req.get("column") or ("msd" if "msd" in (numeric_cols or cols) else ((numeric_cols or cols)[0] if (numeric_cols or cols) else ""))),
                            clearable=False,
                        ),
                        html.Label("window"),
                        dcc.Input(id="util-denoise-window", type="number", value=int(req.get("window", 5)), min=1, step=1),
                        html.Label("group_by"),
                        dcc.Dropdown(
                            id="util-denoise-group",
                            options=col_opts,
                            value=str(req.get("group_by") or ("atom_id" if "atom_id" in cols else (cols[0] if cols else ""))),
                            clearable=False,
                        ),
                        html.Label("x_col"),
                        dcc.Dropdown(
                            id="util-denoise-xcol",
                            options=col_opts,
                            value=str(req.get("x_col") or ("iter" if "iter" in cols else (cols[0] if cols else ""))),
                            clearable=False,
                        ),
                        dcc.Input(id="util-denoise-alpha", type="number", value=0.3, style={"display": "none"}),
                    ]
                )
            else:
                lines.append("No editor for this utility yet.")
            lines.extend(
                [
                    dcc.Input(id="msd-atom-ids", type="text", value="", style={"display": "none"}),
                    dcc.Input(id="msd-atom-types", type="text", value="", style={"display": "none"}),
                    dcc.Input(id="msd-dims", type="text", value="x,y,z", style={"display": "none"}),
                    dcc.Input(id="msd-origin", type="text", value="first", style={"display": "none"}),
                    dcc.Input(id="msd-frames", type="text", value="", style={"display": "none"}),
                    dcc.Input(id="msd-every", type="number", value=1, style={"display": "none"}),
                    dcc.Dropdown(id="util-filter-column", options=col_opts, value=(cols[0] if cols else ""), style={"display": "none"})
                    if util_name not in {"filter_rows", "filter_atoms"}
                    else html.Div(style={"display": "none"}),
                    dcc.Input(id="util-filter-values", type="text", value="", style={"display": "none"})
                    if util_name not in {"filter_rows", "filter_atoms"}
                    else html.Div(style={"display": "none"}),
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
                        dcc.Dropdown(id="viz-x-col", options=opts, value=str(req.get("x_col") or (cols[0] if cols else "")), clearable=False),
                        html.Label("y axis content"),
                        dcc.Dropdown(id="viz-y-col", options=opts, value=str(req.get("y_col") or (cols[0] if cols else "")), clearable=False),
                        html.Label("group content"),
                        dcc.Dropdown(id="viz-group-col", options=opts, value=str(req.get("group_col") or "") or None, clearable=True),
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
                        dcc.Dropdown(id="viz-x-col", options=opts, value=str(req.get("x_col") or (cols[0] if cols else "")), clearable=False),
                        html.Label("y axis content"),
                        dcc.Dropdown(id="viz-y-col", options=opts, value=str(req.get("y_col") or (cols[0] if cols else "")), clearable=False),
                        html.Label("z axis content"),
                        dcc.Dropdown(id="viz-z-col", options=opts, value=str(req.get("z_col") or (cols[0] if cols else "")), clearable=False),
                        html.Label("color by"),
                        dcc.Dropdown(id="viz-color-col", options=opts, value=str(req.get("color_col") or "") or None, clearable=True),
                        dcc.Dropdown(id="viz-group-col", options=opts, value=str(req.get("group_col") or "") or None, clearable=True, style={"display": "none"}),
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
                        dcc.Dropdown(id="viz-x-col", options=opts, value=str(req.get("x_col") or "msd"), clearable=False),
                        dcc.Dropdown(id="viz-y-col", options=opts, value=str(req.get("y_col") or ""), clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-z-col", options=opts, value=str(req.get("z_col") or ""), clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-color-col", options=opts, value=str(req.get("color_col") or "") or None, clearable=True, style={"display": "none"}),
                        dcc.Dropdown(id="viz-group-col", options=opts, value=str(req.get("group_col") or "") or None, clearable=True, style={"display": "none"}),
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
            use_x = str(req.get("x_col") or x_col or "iter")
            use_y = str(req.get("y_col") or y_col or "msd")
            use_group = str(req.get("group_col") or group_col or "")
            use_color = str(req.get("line_color_rgb") or req.get("line_color") or "")
            try:
                use_width = float(req.get("line_width")) if req.get("line_width") is not None else None
            except Exception:
                use_width = None
            use_marker = _parse_float(req.get("marker_size"), 6.0)
            show_markers = _flag_on(req.get("show_markers"), default=False)
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
            for tr in fig.data:
                if not isinstance(tr, go.Scatter):
                    continue
                mode = str(tr.mode or "lines")
                if show_markers and use_marker is not None and use_marker > 0:
                    if "markers" not in mode:
                        mode = f"{mode}+markers" if mode else "markers"
                    tr.update(mode=mode, marker={"size": float(use_marker)})
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
            fig3d = _build_3d(
                rows,
                x_col=str(req.get("x_col") or view3d_x or ""),
                y_col=str(req.get("y_col") or view3d_y or ""),
                z_col=str(req.get("z_col") or view3d_z or ""),
                color_col=str(req.get("color_col") or view3d_color or ""),
            )
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
        graph3d = dcc.Graph(figure=fig3d, config={"displaylogo": False})
        return graph3d, graph3d
