"""Core orchestration for engine resolution + typed data loading + task execution."""

from __future__ import annotations

from pathlib import Path

from reaxkit.core.cache_manager import CacheConfig, CacheManager
from reaxkit.core.engine_registry import resolve_engine
from reaxkit.core.log import get_logger
import reaxkit.engine  # noqa: F401 (register engine adapters)

logger = get_logger(__name__)


class AnalysisExecutor:
    """Orchestrate task execution with strict layer boundaries."""

    def run(self, task, request, args: dict):
        if args.get("verbose"):
            get_logger(__name__, level="DEBUG")
        elif args.get("quiet"):
            get_logger(__name__, level="WARNING")

        input_path = args.get("input") or args.get("run_dir") or "."
        forced_engine = args.get("engine")
        logger.debug("Resolving engine for input=%s forced_engine=%s", input_path, forced_engine)
        adapter = resolve_engine(input_path, engine=forced_engine)
        logger.debug("Resolved adapter=%s", adapter.__class__.__name__)
        data = adapter.load(task.required_data, args)
        logger.debug(
            "Loaded data_type=%s for task=%s",
            getattr(task.required_data, "__name__", str(task.required_data)),
            task.__class__.__name__,
        )

        use_cache = bool(args.get("cache", True)) and not bool(args.get("no_cache", False))
        if not use_cache:
            logger.info("Cache disabled; executing task=%s", task.__class__.__name__)
            return task.run(data, request)

        cache_root = Path(args.get("cache_dir") or (Path(input_path) / ".reaxkit_cache"))
        cache = CacheManager(CacheConfig(root=cache_root, namespace="analysis"))
        key = cache.key_for(task=task, data=data, request=request)

        if cache.exists(key):
            logger.info("Cache hit for task=%s key=%s", task.__class__.__name__, key[:12])
            return cache.load(key)

        logger.info("Cache miss for task=%s key=%s", task.__class__.__name__, key[:12])
        result = task.run(data, request)
        cache.store(key, result)
        logger.debug("Stored result in cache key=%s", key[:12])
        return result
