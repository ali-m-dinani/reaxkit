"""Resolve Web UI runtime output paths outside package source trees.

This module centralizes workspace/cache path defaults for the web UI runtime.
It prevents accidental writes of mutable runtime artifacts under
`reaxkit/src/reaxkit/webui` when the current working directory points there.

**Usage context**

- Workspace defaulting for dataset/session runs.
- Runtime cache directory location for analysis execution.
- Log/trace path composition for UI diagnostics.
"""

from __future__ import annotations

from pathlib import Path


def _webui_source_root() -> Path:
    """Return the package source root for `reaxkit.webui`."""
    return Path(__file__).resolve().parent


def _is_within_webui_source(path: Path) -> bool:
    """Return `True` when `path` is inside the Web UI source tree."""
    src_root = _webui_source_root()
    try:
        path.resolve().relative_to(src_root)
        return True
    except Exception:
        return False


def _safe_dataset_slug(path: Path) -> str:
    """Build a stable folder slug from dataset path."""
    name = str(path.name or "").strip()
    return name if name else "dataset"


def default_workspace_dir_for_dataset(dataset_path: str | None) -> Path:
    """Return a safe default workspace directory for a dataset path.

    If the dataset path resolves inside `reaxkit/src/reaxkit/webui`, the
    workspace is redirected to `~/.reaxkit/workspaces/<dataset>/reaxkit_workspace`
    to avoid mutable artifacts under package source.

    Parameters
    -----
    dataset_path : str | None
        Dataset directory path or file path hint.

    Returns
    -----
    Path
        Absolute workspace directory path.

    Examples
    -----
    ```python
    ws = default_workspace_dir_for_dataset(".")
    ```
    Sample output:
    `Path('.../reaxkit_workspace')`
    Meaning:
    The returned path is safe for logs/cache/output writes.
    """
    base = Path(str(dataset_path or ".").strip() or ".")
    if not base.is_absolute():
        base = (Path.cwd() / base).resolve()
    else:
        base = base.resolve()

    if _is_within_webui_source(base):
        return (Path.home() / ".reaxkit" / "workspaces" / _safe_dataset_slug(base) / "reaxkit_workspace").resolve()
    return (base / "reaxkit_workspace").resolve()


def default_cache_dir_for_workspace(workspace_dir: str | Path | None) -> Path:
    """Return cache directory path rooted under a workspace directory.

    Parameters
    -----
    workspace_dir : str | Path | None
        Workspace root path. Relative values are resolved from current working
        directory.

    Returns
    -----
    Path
        Absolute `.reaxkit_cache` path under the resolved workspace.

    Examples
    -----
    ```python
    cache_dir = default_cache_dir_for_workspace("C:/tmp/reaxkit_workspace")
    ```
    Sample output:
    `Path('C:/tmp/reaxkit_workspace/.reaxkit_cache')`
    Meaning:
    Analysis cache files are isolated under the workspace root.
    """
    root = Path(str(workspace_dir or ".").strip() or ".")
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    else:
        root = root.resolve()
    return (root / ".reaxkit_cache").resolve()
