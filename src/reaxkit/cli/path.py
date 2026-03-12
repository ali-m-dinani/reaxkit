"""
Path utilities and storage maintenance commands for ReaxKit.

This module provides:
- `resolve_output_path(...)` for analysis exports
"""

from __future__ import annotations

from pathlib import Path

from reaxkit.core.storage_layout import ReaxkitStorageLayout, default_project_root, normalize_storage_args

def resolve_output_path(
    user_value: str,
    workflow: str,
    *,
    run_id: str | None = None,
    project_root: str | Path = str(default_project_root()),
    analysis_id: str | None = None,
) -> Path:
    """
    Resolve the output path for a workflow result.

    If the user provides only a bare filename, the file is written under
    ``<project_root>/analysis/<workflow>/<analysis_id_or_run_id>/``.
    If the user provides an absolute path
    or a path containing directories, that path is used directly.
    """
    p = Path(user_value)

    # If user gave an absolute path or a relative path with dirs,
    # respect it exactly.
    if p.is_absolute() or p.parent != Path("."):
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    normalized_root = str(project_root).strip()
    if not normalized_root or normalized_root == ".":
        normalized_root = str(default_project_root())

    norm = normalize_storage_args(
        {
            "run_id": run_id,
            "project_root": normalized_root,
            "analysis_id": analysis_id,
        },
        snapshot=False,
    )
    layout = ReaxkitStorageLayout(project_root=Path(norm["project_root"]))
    layout.ensure_base_layout()
    effective_run_id = str(norm["run_id"])
    scoped = layout.analysis_root / workflow / str(analysis_id or effective_run_id)
    scoped.mkdir(parents=True, exist_ok=True)
    return scoped / p.name
