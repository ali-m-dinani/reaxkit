"""Shell-level callbacks and helpers."""

from __future__ import annotations

import json
import os
import re
import subprocess
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dash import Input, Output, State, ctx, no_update


def default_workspace_dir_for_dataset(dataset_path: str | None) -> str:
    """Build the default workspace path for a dataset directory."""
    base = Path(str(dataset_path or ".").strip() or ".")
    if not base.is_absolute():
        base = (Path.cwd() / base).resolve()
    return str((base / "reaxkit_workspace").resolve())


def _default_repo_slug() -> str:
    explicit = os.environ.get("REAXKIT_GITHUB_REPO", "").strip()
    if explicit:
        return explicit
    try:
        cfg = Path(__file__).resolve().parents[4] / ".git" / "config"
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
            pyproject = Path(__file__).resolve().parents[4] / "pyproject.toml"
            if pyproject.exists():
                text = pyproject.read_text(encoding="utf-8", errors="replace")
                match = re.search(r'^\s*version\s*=\s*"([^"]+)"\s*$', text, re.MULTILINE)
                if match:
                    return str(match.group(1))
        except Exception:
            pass
    return "unknown"


def _version_tuple(raw: str) -> tuple[int, ...]:
    txt = str(raw or "").strip().lstrip("vV")
    parts = re.split(r"[^0-9]+", txt)
    nums = [int(part) for part in parts if part.isdigit()]
    return tuple(nums) if nums else (0,)


def _fetch_latest_release_tag(repo_slug: str) -> tuple[str, str]:
    def _latest_tag_from_git() -> str | None:
        root = Path(__file__).resolve().parents[4]
        commands = [
            ["git", "-C", str(root), "ls-remote", "--tags", "origin"],
            ["git", "ls-remote", "--tags", f"https://github.com/{repo_slug}.git"],
        ]
        tags: set[str] = set()
        for command in commands:
            try:
                proc = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
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
        return sorted(tags, key=lambda tag: (_version_tuple(tag), tag))[-1]

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


def register_shell_callbacks(app, service) -> None:
    """Register shell and top-level navigation callbacks."""

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
                "workspace_dir": default_workspace_dir_for_dataset(os.getcwd()),
                "draft_viz_type": "plot2d",
            },
            "Ready",
        )

    @app.callback(
        Output("ui-store", "data"),
        Input("btn-nav-analysis", "n_clicks"),
        Input("btn-nav-log", "n_clicks"),
        Input("help-menu-trigger", "n_clicks"),
        State("ui-store", "data"),
        prevent_initial_call=True,
    )
    def on_nav_page_change(
        _n_analysis: int,
        _n_log: int,
        _n_help: int,
        ui_data: dict[str, Any] | None,
    ):
        trig = str(ctx.triggered_id or "")
        current = str((ui_data or {}).get("page") or "analysis")
        help_open = bool((ui_data or {}).get("help_open", False))
        if trig == "btn-nav-log":
            return {"page": "log", "help_open": False}
        if trig == "btn-nav-analysis":
            return {"page": "analysis", "help_open": False}
        if trig == "help-menu-trigger":
            return {"page": current, "help_open": not help_open}
        return {"page": current, "help_open": help_open}

    @app.callback(
        Output("panel-left", "style"),
        Output("panel-canvas", "style"),
        Output("panel-props", "style"),
        Output("panel-info", "style"),
        Output("panel-log-page", "style"),
        Output("btn-nav-analysis", "className"),
        Output("btn-nav-log", "className"),
        Output("help-menu-dropdown", "style"),
        Output("help-menu-trigger", "className"),
        Input("ui-store", "data"),
        prevent_initial_call=False,
    )
    def render_active_page(ui_data: dict[str, Any] | None):
        page = str((ui_data or {}).get("page") or "analysis").lower()
        help_open = bool((ui_data or {}).get("help_open", False))
        show_analysis = page == "analysis"
        show_log = page == "log"
        base = "rk-nav-btn"
        return (
            {} if show_analysis else {"display": "none"},
            {} if show_analysis else {"display": "none"},
            {} if show_analysis else {"display": "none"},
            {} if show_analysis else {"display": "none"},
            {"display": "block"} if show_log else {"display": "none"},
            f"{base} active" if show_analysis else base,
            f"{base} active" if show_log else base,
            {"display": "grid"} if help_open else {"display": "none"},
            "rk-help-trigger active" if help_open else "rk-help-trigger",
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
