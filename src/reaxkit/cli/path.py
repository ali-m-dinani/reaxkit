"""
Path resolution helpers for ReaxKit CLI outputs.

This module defines utility logic that maps user-provided output paths to either
explicit filesystem targets or scoped analysis storage locations under the
configured project root.

**Usage context**

- Use these helpers in CLI handlers that accept user export filenames or paths.
- Use scoped storage resolution when a bare filename should be placed under analysis storage.
"""

from __future__ import annotations

from pathlib import Path

from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout, default_project_root, normalize_storage_args

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

    If the user provides a bare filename, the target is scoped under
    ``<project_root>/analysis/<workflow>/<analysis_id_or_run_id>/``. If the user
    provides an absolute path or a relative path that already contains
    directories, that location is used directly and parent directories are
    created as needed.

    Parameters
    -----
    user_value : str
        Output path value provided by the caller.
    workflow : str
        Workflow name used for scoped storage placement.
    run_id : str | None, optional
        Run identifier used when ``analysis_id`` is not provided.
    project_root : str | Path, optional
        Project root used for ReaxKit storage layout resolution.
    analysis_id : str | None, optional
        Analysis identifier that overrides ``run_id`` for scoped storage.

    Returns
    -----
    Path
        Resolved writable output path.

    Examples
    -----
    ```python
    resolve_output_path("result.csv", "bonds", run_id="run-001", project_root="C:/rk")
    ```
    Sample output:
    ```text
    C:/rk/analysis/bonds/run-001/result.csv
    ```
    A bare filename is scoped into the workflow analysis directory.
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
