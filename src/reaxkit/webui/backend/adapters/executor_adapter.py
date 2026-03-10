"""Execution adapter that runs analysis tasks via ReaxKit core executor."""

from __future__ import annotations

import importlib
from dataclasses import fields, is_dataclass
from typing import Any
from typing import get_type_hints


def _build_request(request_type: type, payload: dict[str, Any]) -> object:
    if not is_dataclass(request_type):
        raise TypeError(f"Request type '{request_type}' is not a dataclass")
    valid_fields = {f.name for f in fields(request_type)}
    request_data = {k: v for k, v in payload.items() if k in valid_fields}
    return request_type(**request_data)


def run_analysis_task(task_name: str, request_payload: dict[str, Any], runtime_args: dict[str, Any]) -> tuple[object, type]:
    """Execute a registered task by name and return (result, task_class)."""
    from reaxkit.core.analysis_executor import AnalysisExecutor
    from reaxkit.core.analysis_task_registry import TASK_REGISTRY

    if task_name not in TASK_REGISTRY:
        try:
            from reaxkit.core.analysis_cli_routing_registry import get_registered_analysis_commands

            spec = get_registered_analysis_commands().get(task_name)
            module_path = str(getattr(spec, "module_path", "")).strip() if spec is not None else ""
            if module_path:
                importlib.import_module(module_path)
        except Exception:
            pass

    if task_name not in TASK_REGISTRY:
        raise KeyError(f"Unknown analysis task '{task_name}'. Available: {sorted(TASK_REGISTRY)}")

    task_cls = TASK_REGISTRY[task_name]
    run_params = dict(runtime_args)
    task = task_cls()
    try:
        hints = get_type_hints(task.__class__.run)
    except Exception:
        hints = getattr(task.run, "__annotations__", {}) or {}
    request_type = hints.get("request")
    if request_type is None:
        raise TypeError(f"Task '{task_name}' is missing request type annotation")

    request = _build_request(request_type, request_payload)
    executor = AnalysisExecutor()
    return executor.run(task, request, run_params), task_cls
