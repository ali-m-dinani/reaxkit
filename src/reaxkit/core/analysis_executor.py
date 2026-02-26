"""Core orchestration for engine resolution + typed data loading + task execution."""

from __future__ import annotations

from reaxkit.core.engine_registry import resolve_engine
import reaxkit.engine  # noqa: F401 (register engine adapters)


class AnalysisExecutor:
    """Orchestrate task execution with strict layer boundaries."""

    def run(self, task, request, args: dict):
        input_path = args.get("input") or args.get("run_dir") or "."
        forced_engine = args.get("engine")
        adapter = resolve_engine(input_path, engine=forced_engine)
        data = adapter.load(task.required_data, args)
        return task.run(data, request)
