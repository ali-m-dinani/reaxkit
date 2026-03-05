"""Core orchestration for engine resolution + typed data loading + task execution."""

from __future__ import annotations

import inspect
from pathlib import Path
from time import perf_counter

from reaxkit.core.cache_manager import CacheConfig, CacheManager
from reaxkit.core.engine_registry import resolve_engine
from reaxkit.core.log import get_logger
from reaxkit.core.progress import resolve_reporter
from reaxkit.core.storage_layout import ReaxkitStorageLayout, normalize_storage_args
import reaxkit.engine  # noqa: F401 (register engine adapters)

logger = get_logger(__name__)


class AnalysisExecutor:
    """Orchestrate task execution with strict layer boundaries."""

    DETECTION_HINT_KEYS = (
        "xmolout",
        "geo",
        "fort7",
        "fort13",
        "fort57",
        "fort73",
        "fort74",
        "fort76",
        "fort78",
        "fort79",
        "fort99",
        "summary",
        "trainset",
        "params",
        "control",
        "eregime",
        "vels",
        "molfra",
        "input",
        "run_dir",
    )

    @staticmethod
    def _run_task(task, data, request, reporter):
        params = inspect.signature(task.run).parameters
        if "reporter" in params:
            return task.run(data, request, reporter=reporter)
        return task.run(data, request)

    @classmethod
    def _detection_path(cls, args: dict) -> str:
        for key in cls.DETECTION_HINT_KEYS:
            value = args.get(key)
            if value:
                return str(value)
        return "."

    def run(self, task, request, args: dict):
        normalized = normalize_storage_args(args)
        args.clear()
        args.update(normalized)
        log_level = args.get("log")
        if log_level == "verbose" or args.get("verbose"):
            get_logger(__name__, level="DEBUG")
        elif log_level == "quiet" or args.get("quiet"):
            get_logger(__name__, level="WARNING")

        input_path = self._detection_path(args)
        forced_engine = args.get("engine")
        logger.debug("Resolving engine for input=%s forced_engine=%s", input_path, forced_engine)
        adapter = resolve_engine(input_path, engine=forced_engine)
        logger.debug("Resolved adapter=%s", adapter.__class__.__name__)
        reporter = resolve_reporter(args)
        t_load0 = perf_counter()
        data = adapter.load(task.required_data, args, reporter=reporter)
        run_id = args.get("run_id")
        if run_id:
            try:
                layout = ReaxkitStorageLayout(project_root=Path(args.get("project_root") or "."))
                parser_version = f"{task.__class__.__name__}:{getattr(task.required_data, '__name__', 'data')}"
                engine_name = adapter.__class__.__name__.replace("Adapter", "").lower()
                args["_parsed_id"] = layout.register_parsed_dataset(
                    run_id=str(run_id),
                    parser_version=parser_version,
                    engine=engine_name,
                )
            except Exception as exc:  # pragma: no cover - best-effort metadata persistence
                logger.debug("Storage index update skipped: %s", exc)
        t_load = perf_counter() - t_load0
        logger.debug(
            "Loaded data_type=%s for task=%s",
            getattr(task.required_data, "__name__", str(task.required_data)),
            task.__class__.__name__,
        )
        logger.info("Load time task=%s seconds=%.3f", task.__class__.__name__, t_load)

        use_cache = bool(args.get("cache", True)) and not bool(args.get("no_cache", False))
        if not use_cache:
            logger.info("Cache disabled; executing task=%s", task.__class__.__name__)
            t_run0 = perf_counter()
            result = self._run_task(task, data, request, reporter)
            t_run = perf_counter() - t_run0
            logger.info("Analysis time task=%s seconds=%.3f", task.__class__.__name__, t_run)
            return result

        cache_root = Path(args.get("cache_dir") or (Path(input_path) / ".reaxkit_cache"))
        cache = CacheManager(CacheConfig(root=cache_root, namespace="analysis"))
        key = cache.key_for(task=task, data=data, request=request)

        if cache.exists(key):
            logger.info("Cache hit for task=%s key=%s", task.__class__.__name__, key[:12])
            return cache.load(key)

        logger.info("Cache miss for task=%s key=%s", task.__class__.__name__, key[:12])
        t_run0 = perf_counter()
        result = self._run_task(task, data, request, reporter)
        t_run = perf_counter() - t_run0
        logger.info("Analysis time task=%s seconds=%.3f", task.__class__.__name__, t_run)
        cache.store(key, result)
        logger.debug("Stored result in cache key=%s", key[:12])
        return result
