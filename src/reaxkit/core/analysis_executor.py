"""Core orchestration for engine resolution + typed data loading + task execution."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone
import json

from reaxkit.core.cache_manager import CacheConfig, CacheManager
from reaxkit.core.engine_registry import resolve_engine
from reaxkit.core.exceptions import ParseError, AnalysisError
from reaxkit.core.log import get_logger, configure_file_logging
from reaxkit.core.progress import resolve_reporter
from reaxkit.core.result_time_enrichment import enrich_result_with_time
from reaxkit.core.storage_layout import ReaxkitStorageLayout, normalize_storage_args, snapshot_storage_inputs
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
    def _timing_console_enabled(args: dict) -> bool:
        return bool(args.get("timing") or args.get("show_timing") or args.get("time"))

    @staticmethod
    def _timing_log_path(args: dict) -> Path:
        project_root = Path(args.get("project_root") or ".")
        return project_root / "logs" / "timing" / "machine_readable_timing.log"

    @staticmethod
    def _timing_human_global_log_path(args: dict) -> Path:
        project_root = Path(args.get("project_root") or ".")
        return project_root / "logs" / "timing" / "human_readable_timing.log"

    @staticmethod
    def _timing_human_run_log_path(args: dict) -> Path:
        project_root = Path(args.get("project_root") or ".")
        session_id = str(args.get("_log_session_id") or datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        return project_root / "logs" / "timing" / f"human_readable_timing_{session_id}.log"

    @staticmethod
    def _general_global_log_path(args: dict) -> Path:
        project_root = Path(args.get("project_root") or ".")
        return project_root / "logs" / "general" / "reaxkit_general.log"

    @staticmethod
    def _general_run_log_path(args: dict) -> Path:
        project_root = Path(args.get("project_root") or ".")
        session_id = str(args.get("_log_session_id") or datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        return project_root / "logs" / "general" / f"run_{session_id}.general.log"

    @staticmethod
    def _analysis_output_dir(args: dict) -> Path:
        project_root = Path(args.get("project_root") or ".")
        command = str(args.get("command") or "analysis")
        analysis_id = str(args.get("analysis_id") or args.get("run_id") or args.get("_analysis_id") or "analysis")
        return project_root / "analysis" / command / analysis_id

    @staticmethod
    def _fmt_kv(extra: dict | None) -> str:
        if not extra:
            return ""
        parts: list[str] = []
        for key in sorted(extra.keys()):
            value = extra.get(key)
            if value is None:
                continue
            parts.append(f"{key}={value}")
        return (" " + " ".join(parts)) if parts else ""

    @classmethod
    def _record_general(cls, args: dict, *, event: str, task_name: str, extra: dict | None = None) -> None:
        stamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        line = (
            f"{stamp} ReaxKit event={event} task={task_name} "
            f"run_id={args.get('run_id', '')}"
            f"{cls._fmt_kv(extra)}"
        )
        for path in (cls._general_global_log_path(args), cls._general_run_log_path(args)):
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    @classmethod
    def _record_timing(cls, args: dict, *, phase: str, task_name: str, seconds: float, extra: dict | None = None) -> None:
        payload = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "phase": str(phase),
            "task": str(task_name),
            "seconds": float(seconds),
            "run_id": args.get("run_id"),
            "session_id": args.get("_log_session_id"),
        }
        if extra:
            payload.update(extra)

        path = cls._timing_log_path(args)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")

        human_line = (
            f"{datetime.now().strftime('%m-%d-%Y-%H-%M-%S')} "
            f"ReaxKit task={task_name} phase={phase} "
            f"time={float(seconds):.3f}s run_id={args.get('run_id', '')}"
            f"{cls._fmt_kv(extra)}"
        )
        for human_path in (cls._timing_human_global_log_path(args), cls._timing_human_run_log_path(args)):
            human_path.parent.mkdir(parents=True, exist_ok=True)
            with open(human_path, "a", encoding="utf-8") as fh:
                fh.write(human_line + "\n")

    @classmethod
    def _load_timing_callback(cls, args: dict, *, task_name: str):
        def _emit(*, handler: str, source: str | None, source_path: str | None, seconds: float) -> None:
            cls._record_timing(
                args,
                phase="load_handler",
                task_name=task_name,
                seconds=float(seconds),
                extra={
                    "handler": handler,
                    "source": source,
                    "source_path": source_path,
                },
            )

        return _emit

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
        normalized = normalize_storage_args(args, snapshot=False)
        args.clear()
        args.update(normalized)
        handler_cache_dir = Path(args.get("project_root") or ".") / "cache" / "handlers"
        handler_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["REAXKIT_HANDLER_CACHE_DIR"] = str(handler_cache_dir.resolve())
        session_id = configure_file_logging(Path(args.get("project_root") or "."))
        args["_log_session_id"] = session_id
        args["_load_timing_callback"] = self._load_timing_callback(args, task_name=task.__class__.__name__)
        log_level = args.get("log")
        if log_level == "verbose" or args.get("verbose"):
            get_logger(__name__, level="DEBUG")
        elif log_level == "quiet" or args.get("quiet"):
            get_logger(__name__, level="WARNING")
        self._record_general(
            args,
            event="session_start",
            task_name=task.__class__.__name__,
            extra={
                "command": args.get("command"),
                "project_root": args.get("project_root"),
            },
        )

        input_path = str(args.get("_snapshot_source_dir") or self._detection_path(args))
        forced_engine = args.get("engine")
        logger.debug("Resolving engine for input=%s forced_engine=%s", input_path, forced_engine)
        adapter = resolve_engine(input_path, engine=forced_engine)
        logger.debug("Resolved adapter=%s", adapter.__class__.__name__)
        snapshot_names = adapter.required_input_files(task.required_data, args)
        snapshot_storage_inputs(args, names=snapshot_names)
        if run_id := args.get("run_id"):
            project_root = Path(args.get("project_root") or ".")
            raw_dir = ReaxkitStorageLayout(project_root=project_root).raw_run_dir(str(run_id))
            copied = sorted([p.name for p in raw_dir.iterdir() if p.is_file()]) if raw_dir.exists() else []
            self._record_general(
                args,
                event="snapshot_raw_ready",
                task_name=task.__class__.__name__,
                extra={
                    "raw_dir": str(raw_dir),
                    "files": ",".join(copied),
                    "required_inputs": ",".join(snapshot_names or ()),
                    "engine_adapter": adapter.__class__.__name__,
                },
            )
        reporter = resolve_reporter(args)
        run_id = args.get("run_id")
        task_version = str(getattr(task, "VERSION", "1"))
        layout = ReaxkitStorageLayout(project_root=Path(args.get("project_root") or ".")) if run_id else None
        parsed_id = None
        if layout is not None:
            try:
                handler_version = str(getattr(adapter, "HANDLER_VERSION", "1"))
                engine_name = adapter.__class__.__name__.replace("Adapter", "").lower()
                args["_parsed_id"] = layout.register_parsed_dataset(
                    run_id=str(run_id),
                    handler_version=handler_version,
                    engine=engine_name,
                )
                parsed_id = str(args["_parsed_id"])
                self._record_general(
                    args,
                    event="parsed_dataset_registered",
                    task_name=task.__class__.__name__,
                    extra={
                        "parsed_id": parsed_id,
                        "parsed_dir": str(layout.parsed_dir(parsed_id)),
                        "run_index": str(layout.run_index_path(str(run_id))),
                    },
                )
            except Exception as exc:  # pragma: no cover - best-effort metadata persistence
                logger.debug("Storage index update skipped: %s", exc)

        cache_root = Path(args.get("cache_dir") or (Path(input_path) / ".reaxkit_cache"))
        cache = CacheManager(CacheConfig(root=cache_root, namespace="analysis"))
        use_cache = bool(args.get("cache", True)) and not bool(args.get("no_cache", False))
        analysis_id = None
        if parsed_id is not None:
            data_name = getattr(task.required_data, "__name__", "parsed_data")
            artifact_name = str(data_name).lower()
            if layout is not None:
                cached_parsed = layout.load_parsed_artifact(
                    parsed_id=parsed_id,
                    artifact_name=artifact_name,
                )
                if cached_parsed is not None:
                    expected_type = task.required_data
                    if isinstance(cached_parsed, expected_type):
                        logger.debug("Parsed cache hit for data_type=%s parsed_id=%s", data_name, parsed_id[:12])
                        t_load = 0.0
                        self._record_timing(args, phase="load_total", task_name=task.__class__.__name__, seconds=t_load)
                        data = cached_parsed
                        analysis_id = cache.analysis_id_for(
                            task=task,
                            data=None,
                            request=request,
                            parsed_id=parsed_id,
                            task_version=task_version,
                        )
                        args["_analysis_id"] = analysis_id
                        try:
                            layout.record_run_analysis(
                                run_id=str(run_id),
                                parsed_id=parsed_id,
                                analysis_id=analysis_id,
                                task_name=task.__class__.__name__,
                                task_version=task_version,
                            )
                        except Exception as exc:  # pragma: no cover - best-effort index update
                            logger.debug("Run analysis index update skipped: %s", exc)

                        if use_cache and cache.exists(analysis_id):
                            logger.info("Cache hit for task=%s analysis_id=%s", task.__class__.__name__, analysis_id[:12])
                            self._record_general(
                                args,
                                event="analysis_cache_hit",
                                task_name=task.__class__.__name__,
                                extra={"analysis_id": analysis_id},
                            )
                            cached = cache.load(analysis_id)
                            return enrich_result_with_time(
                                cached,
                                data,
                                control_file=str(args.get("control") or "control"),
                            )

                        if not use_cache:
                            logger.info("Cache disabled; executing task=%s", task.__class__.__name__)
                        else:
                            logger.info("Cache miss for task=%s analysis_id=%s", task.__class__.__name__, analysis_id[:12])
                            self._record_general(
                                args,
                                event="analysis_cache_miss",
                                task_name=task.__class__.__name__,
                                extra={"analysis_id": analysis_id},
                            )
                        t_run0 = perf_counter()
                        try:
                            result = self._run_task(task, data, request, reporter)
                        except AnalysisError:
                            raise
                        except Exception as exc:
                            raise AnalysisError(
                                f"Task '{task.__class__.__name__}' failed during analysis: {exc}"
                            ) from exc
                        result = enrich_result_with_time(
                            result,
                            data,
                            control_file=str(args.get("control") or "control"),
                        )
                        t_run = perf_counter() - t_run0
                        self._record_timing(args, phase="analyze", task_name=task.__class__.__name__, seconds=t_run)
                        if use_cache:
                            cache.store(analysis_id, result, task_name=task.__class__.__name__)
                            logger.debug("Stored result in cache analysis_id=%s", analysis_id[:12])
                        self._record_general(
                            args,
                            event="analysis_done",
                            task_name=task.__class__.__name__,
                            extra={
                                "analysis_id": analysis_id,
                                "analysis_dir": str(self._analysis_output_dir(args)),
                                "status": "success",
                            },
                        )
                        return result

            analysis_id = cache.analysis_id_for(
                task=task,
                data=None,
                request=request,
                parsed_id=parsed_id,
                task_version=task_version,
            )
            args["_analysis_id"] = analysis_id
            try:
                layout.record_run_analysis(
                    run_id=str(run_id),
                    parsed_id=parsed_id,
                    analysis_id=analysis_id,
                    task_name=task.__class__.__name__,
                    task_version=task_version,
                )
            except Exception as exc:  # pragma: no cover - best-effort index update
                logger.debug("Run analysis index update skipped: %s", exc)

            if use_cache and cache.exists(analysis_id):
                logger.info("Cache hit for task=%s analysis_id=%s", task.__class__.__name__, analysis_id[:12])
                self._record_general(
                    args,
                    event="analysis_cache_hit",
                    task_name=task.__class__.__name__,
                    extra={"analysis_id": analysis_id},
                )
                cached = cache.load(analysis_id)
                return enrich_result_with_time(
                    cached,
                    None,
                    control_file=str(args.get("control") or "control"),
                )

        t_load0 = perf_counter()
        try:
            data = adapter.load(task.required_data, args, reporter=reporter)
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError(
                f"Failed to load required data '{getattr(task.required_data, '__name__', str(task.required_data))}' "
                f"for task '{task.__class__.__name__}': {exc}"
            ) from exc
        if run_id and layout is not None and parsed_id is None:
            try:
                handler_version = str(getattr(adapter, "HANDLER_VERSION", "1"))
                engine_name = adapter.__class__.__name__.replace("Adapter", "").lower()
                args["_parsed_id"] = layout.register_parsed_dataset(
                    run_id=str(run_id),
                    handler_version=handler_version,
                    engine=engine_name,
                )
                parsed_id = str(args["_parsed_id"])
                self._record_general(
                    args,
                    event="parsed_dataset_registered",
                    task_name=task.__class__.__name__,
                    extra={
                        "parsed_id": parsed_id,
                        "parsed_dir": str(layout.parsed_dir(parsed_id)),
                        "run_index": str(layout.run_index_path(str(run_id))),
                    },
                )
            except Exception as exc:  # pragma: no cover - best-effort metadata persistence
                logger.debug("Storage index update skipped: %s", exc)

        if parsed_id is not None and layout is not None:
            try:
                data_name = getattr(task.required_data, "__name__", "parsed_data")
                artifact_name = data_name.lower()
                parsed_artifact_path = layout.persist_parsed_artifact(
                    parsed_id=parsed_id,
                    artifact_name=artifact_name,
                    data=data,
                )
                self._record_general(
                    args,
                    event="parsed_artifact_saved",
                    task_name=task.__class__.__name__,
                    extra={
                        "parsed_id": parsed_id,
                        "artifact": artifact_name,
                        "path": str(parsed_artifact_path),
                    },
                )
            except Exception as exc:  # pragma: no cover - best-effort artifact persistence
                logger.debug("Parsed artifact write skipped: %s", exc)
        t_load = perf_counter() - t_load0
        logger.debug(
            "Loaded data_type=%s for task=%s",
            getattr(task.required_data, "__name__", str(task.required_data)),
            task.__class__.__name__,
        )
        self._record_timing(args, phase="load_total", task_name=task.__class__.__name__, seconds=t_load)

        analysis_id = analysis_id or cache.analysis_id_for(
            task=task,
            data=data,
            request=request,
            parsed_id=parsed_id,
            task_version=task_version,
        )
        args["_analysis_id"] = analysis_id
        if run_id:
            try:
                layout.record_run_analysis(
                    run_id=str(run_id),
                    parsed_id=parsed_id,
                    analysis_id=analysis_id,
                    task_name=task.__class__.__name__,
                    task_version=task_version,
                )
            except Exception as exc:  # pragma: no cover - best-effort index update
                logger.debug("Run analysis index update skipped: %s", exc)

        if not use_cache:
            logger.info("Cache disabled; executing task=%s", task.__class__.__name__)
            t_run0 = perf_counter()
            try:
                result = self._run_task(task, data, request, reporter)
            except AnalysisError:
                raise
            except Exception as exc:
                raise AnalysisError(
                    f"Task '{task.__class__.__name__}' failed during analysis: {exc}"
                ) from exc
            result = enrich_result_with_time(
                result,
                data,
                control_file=str(args.get("control") or "control"),
            )
            t_run = perf_counter() - t_run0
            self._record_timing(args, phase="analyze", task_name=task.__class__.__name__, seconds=t_run)
            self._record_general(
                args,
                event="analysis_done",
                task_name=task.__class__.__name__,
                extra={
                    "analysis_id": analysis_id,
                    "analysis_dir": str(self._analysis_output_dir(args)),
                    "status": "success",
                },
            )
            return result

        if cache.exists(analysis_id):
            logger.info("Cache hit for task=%s analysis_id=%s", task.__class__.__name__, analysis_id[:12])
            self._record_general(
                args,
                event="analysis_cache_hit",
                task_name=task.__class__.__name__,
                extra={"analysis_id": analysis_id},
            )
            cached = cache.load(analysis_id)
            return enrich_result_with_time(
                cached,
                data,
                control_file=str(args.get("control") or "control"),
            )

        logger.info("Cache miss for task=%s analysis_id=%s", task.__class__.__name__, analysis_id[:12])
        self._record_general(
            args,
            event="analysis_cache_miss",
            task_name=task.__class__.__name__,
            extra={"analysis_id": analysis_id},
        )
        t_run0 = perf_counter()
        try:
            result = self._run_task(task, data, request, reporter)
        except AnalysisError:
            raise
        except Exception as exc:
            raise AnalysisError(
                f"Task '{task.__class__.__name__}' failed during analysis: {exc}"
            ) from exc
        result = enrich_result_with_time(
            result,
            data,
            control_file=str(args.get("control") or "control"),
        )
        t_run = perf_counter() - t_run0
        self._record_timing(args, phase="analyze", task_name=task.__class__.__name__, seconds=t_run)
        cache.store(analysis_id, result, task_name=task.__class__.__name__)
        logger.debug("Stored result in cache analysis_id=%s", analysis_id[:12])
        self._record_general(
            args,
            event="analysis_done",
            task_name=task.__class__.__name__,
            extra={
                "analysis_id": analysis_id,
                "analysis_dir": str(self._analysis_output_dir(args)),
                "status": "success",
            },
        )
        return result
