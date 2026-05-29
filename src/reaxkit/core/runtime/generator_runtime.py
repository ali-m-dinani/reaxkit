"""Run-scoped output helpers for generator commands."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reaxkit.core.platform.log import configure_file_logging
from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout, default_project_root, generate_run_id


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def prepare_generator_output(args: Any, *, command: str, output_value: str) -> tuple[Path, ReaxkitStorageLayout]:
    """Resolve generator output to inputs/<run_id>/<filename> and normalize args in-place."""
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
    """Persist per-run generator settings JSON and update run index."""
    run_id = str(getattr(args, "run_id"))
    settings_path = output_path.parent / "settings.json"
    payload: dict[str, Any] = {
        "command": str(command),
        "run_id": run_id,
        "project_root": str(getattr(args, "project_root", "")),
        "output": {
            "name": output_path.name,
            "path": str(output_path),
        },
        "copy_to_dot": bool(copy_to_dot),
        "args": _json_safe(vars(args)),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload["extra"] = _json_safe(extra)
    settings_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    layout.record_run_generator(
        run_id=run_id,
        command=command,
        output_path=output_path,
        settings_path=settings_path,
    )
    return settings_path


def maybe_copy_output_to_dot(output_path: Path, *, enabled: bool) -> Path | None:
    """Copy generated output to current directory when requested."""
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
    seen: set[str] = set()
    print("Results saved in:")
    for path in dirs:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        print(f"  {resolved}")
