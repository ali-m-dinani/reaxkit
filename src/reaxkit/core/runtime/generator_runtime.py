"""
Run-scoped output helpers for generator commands.

**Usage context**

- Import these helpers from ReaxKit core modules when implementing CLI and workflow logic.
- Reuse the public APIs here to keep behavior consistent across commands and engines.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reaxkit.core.platform.log import configure_file_logging
from reaxkit.core.runtime.provenance import (
    effective_settings_from_args,
    json_safe,
    runtime_metadata_from_args,
    user_settings_from_args,
)
from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout, default_project_root, generate_run_id


def prepare_generator_output(args: Any, *, command: str, output_value: str) -> tuple[Path, ReaxkitStorageLayout]:
    """
    Resolve generator output to inputs/<run_id>/<filename> and normalize args in-place.
    
    This function is part of the ReaxKit core API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    args : Any
        Input parameter used by this function.
    command : str
        Input parameter used by this function.
    output_value : str
        Input parameter used by this function.
    
    Returns
    -----
    tuple[Path, ReaxkitStorageLayout]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.runtime.generator_runtime import prepare_generator_output
    # Configure required arguments for your case.
    result = prepare_generator_output(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    run_id = str(getattr(args, "run_id", None) or generate_run_id())
    project_root = Path(getattr(args, "project_root", None) or default_project_root())

    setattr(args, "run_id", run_id)
    setattr(args, "project_root", str(project_root))
    configure_file_logging(project_root)

    layout = ReaxkitStorageLayout(project_root=project_root)
    layout.ensure_input_run_layout(run_id)

    requested = Path(str(output_value or "output"))
    filename = requested.name or "output"
    out_path = layout.input_run_dir(run_id) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path, layout


def persist_generator_metadata(
    args: Any,
    *,
    command: str,
    output_path: Path,
    layout: ReaxkitStorageLayout,
    extra: dict[str, Any] | None = None,
    copy_to_dot: bool = False,
) -> Path:
    """
    Persist per-run generator settings JSON and update run index.
    
    This function is part of the ReaxKit core API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    args : Any
        Input parameter used by this function.
    command : str
        Input parameter used by this function.
    output_path : Path
        Input parameter used by this function.
    layout : ReaxkitStorageLayout
        Input parameter used by this function.
    extra : dict[str, Any] | None, optional
        Input parameter used by this function.
    copy_to_dot : bool, optional
        Input parameter used by this function.
    
    Returns
    -----
    Path
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.runtime.generator_runtime import persist_generator_metadata
    # Configure required arguments for your case.
    result = persist_generator_metadata(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    run_id = str(getattr(args, "run_id"))
    settings_path = output_path.parent / "settings.json"
    user_settings = user_settings_from_args(args)
    effective_settings = effective_settings_from_args(args)
    payload: dict[str, Any] = {
        "command": str(command),
        "run_id": run_id,
        "project_root": str(getattr(args, "project_root", "")),
        "user_settings": user_settings,
        "effective_settings": effective_settings,
        "runtime": runtime_metadata_from_args(args),
        "output": {
            "name": output_path.name,
            "path": str(output_path),
        },
        "copy_to_dot": bool(copy_to_dot),
        "args": json_safe(vars(args)),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload["extra"] = json_safe(extra)
    settings_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    layout.record_run_generator(
        run_id=run_id,
        command=command,
        output_path=output_path,
        settings_path=settings_path,
        user_settings=user_settings,
    )
    return settings_path


def maybe_copy_output_to_dot(output_path: Path, *, enabled: bool) -> Path | None:
    """
    Copy generated output to current directory when requested.
    
    This function is part of the ReaxKit core API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    output_path : Path
        Input parameter used by this function.
    enabled : bool
        Input parameter used by this function.
    
    Returns
    -----
    Path | None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.runtime.generator_runtime import maybe_copy_output_to_dot
    # Configure required arguments for your case.
    result = maybe_copy_output_to_dot(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    if not enabled:
        return None
    dst = Path("..") / output_path.name
    if output_path.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(output_path, dst)
    else:
        shutil.copy2(output_path, dst)
    return dst


def print_saved_dirs(dirs: list[Path]) -> None:
    """
    Print saved dirs.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    dirs : list[Path]
        Input parameter used by this function.
    
    Returns
    -----
    None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.runtime.generator_runtime import print_saved_dirs
    # Configure required arguments for your case.
    result = print_saved_dirs(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    seen: set[str] = set()
    print("Results saved in:")
    for path in dirs:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        print(f"  {resolved}")
