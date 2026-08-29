"""
Core orchestration for engine resolution + typed data loading + task execution.

**Usage context**

- Import these helpers from ReaxKit core modules when implementing CLI and workflow logic.
- Reuse the public APIs here to keep behavior consistent across commands and engines.
"""

from __future__ import annotations

import copy
import inspect
import os
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from reaxkit.core.storage.cache_manager import CacheConfig, CacheManager
from reaxkit.core.platform.engine_resolver import resolve_engine
from reaxkit.core.platform.exceptions import ParseError, AnalysisError
from reaxkit.core.platform.log import get_logger, configure_file_logging
from reaxkit.core.runtime.progress import progress_operation, resolve_reporter
from reaxkit.core.runtime.provenance import user_settings_from_args
from reaxkit.core.results_shaping.result_time_enrichment import enrich_result_with_time
from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout, normalize_storage_args, snapshot_storage_inputs
import reaxkit.engine  # noqa: F401 (register engine adapters)

logger = get_logger(__name__)


class AnalysisExecutor:
    """Orchestrate task execution with strict layer boundaries."""

    FRAME_SELECTIVE_DATA_TYPES = {
        "TrajectoryData",
        "ChargeData",
        "ConnectivityData",
        "ConnectivityTrajectoryData",
        "CoordinationStatusBundleData",
        "ElectrostaticsData",
    }

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
        "dump",
        "rkf",
        "ams",
        "trajectory",
        "connectivity",
        "charges",
        "electric_field",
        "input",
        "run_dir",
    )

    @staticmethod
    def _timing_console_enabled(args: dict) -> bool:
        """
        Timing console enabled.
        """
        return bool(args.get("timing") or args.get("show_timing") or args.get("time"))

    @staticmethod
    def _timing_log_path(args: dict) -> Path:
        """
        Timing log path.
        """
        project_root = Path(args.get("project_root") or ".")
        return project_root / "logs" / "timing" / "machine_readable_timing.log"

    @staticmethod
    def _timing_human_global_log_path(args: dict) -> Path:
        """
        Timing human global log path.
        """
        project_root = Path(args.get("project_root") or ".")
        return project_root / "logs" / "timing" / "human_readable_timing.log"

    @staticmethod
    def _timing_human_run_log_path(args: dict) -> Path:
        """
        Timing human run log path.
        """
        project_root = Path(args.get("project_root") or ".")
        session_id = str(args.get("_log_session_id") or datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        return project_root / "logs" / "timing" / f"human_readable_timing_{session_id}.log"

    @staticmethod
    def _general_global_log_path(args: dict) -> Path:
        """
        General global log path.
        """
        project_root = Path(args.get("project_root") or ".")
        return project_root / "logs" / "general" / "reaxkit_general.log"

    @staticmethod
    def _general_run_log_path(args: dict) -> Path:
        """
        General run log path.
        """
        project_root = Path(args.get("project_root") or ".")
        session_id = str(args.get("_log_session_id") or datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        return project_root / "logs" / "general" / f"run_{session_id}.general.log"

    @staticmethod
    def _analysis_output_dir(args: dict) -> Path:
        """
        Analysis output dir.
        """
        project_root = Path(args.get("project_root") or ".")
        command = str(args.get("command") or "analysis")
        analysis_id = str(args.get("analysis_id") or args.get("run_id") or args.get("_analysis_id") or "analysis")
        return project_root / "analysis" / command / analysis_id

    @staticmethod
    def _fmt_kv(extra: dict | None) -> str:
        """
        Fmt kv.
        """
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
        """
        Record general.
        """
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
        """
        Record timing.
        """
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
        """
        Load timing callback.
        """
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

    @classmethod
    def _requested_frame_indices(cls, request, required_data) -> list[int] | None:
        """Return explicit source-frame dependencies for a partial load."""
        if getattr(required_data, "__name__", "") not in cls.FRAME_SELECTIVE_DATA_TYPES:
            return None

        primary = None
        for name in ("frames", "frame_indices", "frame"):
            value = getattr(request, name, None)
            if value is None or isinstance(value, (str, bytes)):
                continue
            try:
                values = [int(value)] if np.isscalar(value) else [int(i) for i in value]
            except TypeError:
                continue
            if values:
                primary = values
                break
        if not primary:
            return None

        # Some analyses need an additional reference snapshot that is not in
        # the main frame list (for example displacement relative to frame 0).
        dependencies = list(primary)
        for name in ("reference_frame",):
            value = getattr(request, name, None)
            if value is not None:
                dependencies.append(int(value))
        return list(dict.fromkeys(i for i in dependencies if i >= 0))

    @staticmethod
    def _source_frame_indices(data) -> list[int] | None:
        """Find the compact-frame to source-frame mapping on loaded data."""
        candidates = [data, getattr(data, "trajectory", None), getattr(data, "connectivity", None)]
        for candidate in candidates:
            if candidate is None:
                continue
            values = getattr(candidate, "source_frame_indices", None)
            if values is None:
                metadata = getattr(candidate, "metadata", None)
                if isinstance(metadata, dict):
                    values = metadata.get("source_frame_indices")
            if values is None:
                continue
            return [int(i) for i in np.asarray(values, dtype=int).reshape(-1)]
        return None

    @staticmethod
    def _use_selective_sources(args: dict, snapshot_names) -> tuple[tuple[str, ...], dict[str, str]]:
        """Keep large partial-load sources in place instead of copying them."""
        source_dir = Path(str(args.get("_snapshot_source_dir") or "."))
        borrowed: dict[str, str] = {}
        remaining: list[str] = []
        for name in tuple(snapshot_names or ()):
            name_s = str(name)
            name_lower = name_s.lower()
            if name_lower == "xmolout":
                key = "xmolout"
            elif name_lower == "fort.7":
                key = "fort7"
            elif Path(name_s).suffix.lower() in {".kf", ".rkf"}:
                key = "rkf"
            elif "dump" in name_lower or "lammpstrj" in name_lower:
                key = "dump"
            else:
                key = None
            explicit = Path(str(args.get(key))) if key is not None and args.get(key) else None
            candidate = explicit if explicit is not None and explicit.is_file() else source_dir / str(name)
            if key is not None and candidate.is_file():
                resolved = str(candidate.resolve())
                args[key] = resolved
                borrowed[str(name)] = resolved
            else:
                remaining.append(str(name))
        if borrowed:
            args["_selective_source_files"] = borrowed
        return tuple(remaining), borrowed

    @classmethod
    def _request_for_loaded_frames(cls, request, data):
        """Translate source frame selectors to compact in-memory indices."""
        source_indices = cls._source_frame_indices(data)
        if source_indices is None:
            return request, None
        source_to_local = {source: local for local, source in enumerate(source_indices)}
        execution_request = copy.copy(request)

        for name in ("frames", "frame_indices"):
            values = getattr(request, name, None)
            if values is None or isinstance(values, (str, bytes)):
                continue
            setattr(
                execution_request,
                name,
                [source_to_local[int(i)] for i in values if int(i) in source_to_local],
            )
        if hasattr(request, "frame"):
            frame_value = getattr(request, "frame")
            if frame_value is not None and int(frame_value) in source_to_local:
                setattr(execution_request, "frame", source_to_local[int(frame_value)])
        if hasattr(request, "reference_frame"):
            reference_value = getattr(request, "reference_frame")
            if reference_value is not None:
                source_reference = int(reference_value)
                if source_reference in source_to_local:
                    setattr(execution_request, "reference_frame", source_to_local[source_reference])
        return execution_request, source_indices

    @staticmethod
    def _restore_result_frame_indices(result, source_indices: list[int] | None, original_request):
        """Restore user-facing source indices after compact-frame analysis."""
        if source_indices is None:
            return result
        local_to_source = {local: source for local, source in enumerate(source_indices)}

        def _restore_frame(frame: pd.DataFrame) -> pd.DataFrame:
            columns = [name for name in ("frame_index", "frame_idx") if name in frame.columns]
            if not columns:
                return frame
            out = frame.copy()
            for name in columns:
                numeric = pd.to_numeric(out[name], errors="coerce")
                out[name] = [
                    local_to_source.get(int(value), value) if pd.notna(value) else value
                    for value in numeric
                ]
            return out

        if isinstance(result, pd.DataFrame):
            result = _restore_frame(result)
        elif hasattr(result, "__dict__"):
            for name, value in vars(result).items():
                if isinstance(value, pd.DataFrame):
                    setattr(result, name, _restore_frame(value))
            if hasattr(result, "request"):
                result.request = original_request
        return result

    @classmethod
    def _run_task(cls, task, data, request, reporter):
        """
        Run task.
        """
        execution_request, source_indices = cls._request_for_loaded_frames(request, data)
        task_name = task.__class__.__name__
        with progress_operation(
            reporter,
            "analyze",
            f"Running {task_name}",
            f"Finished {task_name}",
        ) as analysis_reporter:
            params = inspect.signature(task.run).parameters
            if "reporter" in params:
                result = task.run(data, execution_request, reporter=analysis_reporter)
            else:
                result = task.run(data, execution_request)
        return cls._restore_result_frame_indices(result, source_indices, request)

    @staticmethod
    def _streaming_enabled(task, adapter, required_data, args: dict, requested_frame_indices) -> bool:
        """Select streaming for all-frame tasks with explicit opt-in support."""
        return bool(
            args.get("stream", True)
            and requested_frame_indices is None
            and callable(getattr(task, "run_stream", None))
            and adapter.supports_streaming(required_data, args)
        )

    @classmethod
    def _run_stream_task(cls, task, frames, request, reporter):
        """Execute an incremental task against a canonical frame iterator."""
        task_name = task.__class__.__name__
        with progress_operation(
            reporter,
            "stream",
            f"Streaming {task_name}",
            f"Finished {task_name}",
        ) as stream_reporter:
            params = inspect.signature(task.run_stream).parameters
            if "reporter" in params:
                return task.run_stream(frames, request, reporter=stream_reporter)
            return task.run_stream(frames, request)

    @classmethod
    def _stream_source_identity(cls, adapter, required_data, args: dict, required_names) -> dict:
        """Build a cache identity from source paths without reading file data."""
        candidates: list[Path] = []
        for key in cls.DETECTION_HINT_KEYS:
            value = args.get(key)
            if not value:
                continue
            path = Path(str(value))
            if path.is_file():
                candidates.append(path)
            elif path.is_dir():
                candidates.extend(path / str(name) for name in tuple(required_names or ()))
        for value in dict(args.get("_selective_source_files") or {}).values():
            candidates.append(Path(str(value)))
        source_dir = Path(str(args.get("_snapshot_source_dir") or args.get("run_dir") or "."))
        candidates.extend(source_dir / str(name) for name in tuple(required_names or ()))

        sources: list[dict[str, object]] = []
        seen: set[str] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                if not resolved.is_file() or str(resolved) in seen:
                    continue
                seen.add(str(resolved))
                stat = resolved.stat()
                sources.append(
                    {
                        "path": str(resolved),
                        "size": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                    }
                )
            except OSError:
                continue
        return {
            "adapter": adapter.__class__.__name__,
            "data_type": getattr(required_data, "__name__", str(required_data)),
            "sources": sorted(sources, key=lambda item: str(item["path"])),
            "cwd": str(Path.cwd().resolve()),
            "streaming": True,
        }

    @classmethod
    def _detection_path(cls, args: dict) -> str:
        """
        Detection path.
        """
        for key in cls.DETECTION_HINT_KEYS:
            value = args.get(key)
            if value:
                return str(value)
        return "."

    @classmethod
    def _engine_detection_path(cls, args: dict) -> str:
        """
        Return the path used for engine detection.

        An explicit --input must win over the snapshot source directory. The
        snapshot source is often the parent directory of a file input, and that
        directory can contain mixed engine artifacts.
        """
        explicit_input = args.get("input")
        input_was_explicit = args.get("_input_was_explicit")
        if input_was_explicit is None:
            # Support callers that have not passed through storage argument
            # normalization, where a non-default input is necessarily explicit.
            input_was_explicit = bool(explicit_input and str(explicit_input) != ".")
        if input_was_explicit and explicit_input and str(explicit_input) != ".":
            return str(explicit_input)
        return str(args.get("_snapshot_source_dir") or cls._detection_path(args))

    @staticmethod
    def _console_step(args: dict, message: str) -> None:
        """
        Console step.
        """
        if args.get("quiet") or not args.get("log_in_terminal"):
            return
        run_id = args.get("run_id")
        suffix = f" run_id={run_id}" if run_id else ""
        print(f"[ReaxKit] {message}{suffix}", flush=True)

    def run(self, task, request, args: dict):
        # ---------------------------------------------------------------------
        # 1) Normalize runtime/storage arguments and derive task/data metadata.
        # ---------------------------------------------------------------------
        """
        Run.
        
        This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
        
        Parameters
        -----
        task : Any
            Input parameter used by this function.
        request : Any
            Input parameter used by this function.
        args : dict
            Input parameter used by this function.
        
        Returns
        -----
        Any
            Value produced by this function call.
        
        Examples
        -----
        ```python
        from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
        instance = AnalysisExecutor(...)
        # Configure required arguments for your case.
        result = instance.run(...)
        print(type(result).__name__)
        ```
        Sample output:
        ```text
        str
        ```
        The output type reflects the return contract for this API call.
        """
        normalized = normalize_storage_args(args, snapshot=False)
        args.clear()
        args.update(normalized)
        required_data = (
            task.required_data_for(request, args) if hasattr(task, "required_data_for") else getattr(task, "required_data", None)
        )
        requested_frame_indices = self._requested_frame_indices(request, required_data)
        if requested_frame_indices is not None:
            args["_frame_indices"] = requested_frame_indices
        task_name = task.__class__.__name__
        data_name = getattr(required_data, "__name__", str(required_data))

        # ---------------------------------------------------------------------
        # 2) Initialize runtime side effects (terminal logs, handler cache env,
        #    file logging session, timing hooks, and log verbosity).
        # ---------------------------------------------------------------------
        self._console_step(args, f"Starting task={task_name} required_data={data_name}")
        handler_cache_dir = Path(args.get("project_root") or ".") / "cache" / "handlers"
        handler_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["REAXKIT_HANDLER_CACHE_DIR"] = str(handler_cache_dir.resolve())
        self._console_step(args, f"Handler cache dir={handler_cache_dir}")
        session_id = configure_file_logging(Path(args.get("project_root") or "."))
        args["_log_session_id"] = session_id
        args["_load_timing_callback"] = self._load_timing_callback(args, task_name=task_name)
        log_level = args.get("log")
        if log_level == "verbose" or args.get("verbose"):
            get_logger(__name__, level="DEBUG")
        elif log_level == "quiet" or args.get("quiet"):
            get_logger(__name__, level="WARNING")
        self._record_general(
            args,
            event="session_start",
            task_name=task_name,
            extra={
                "command": args.get("command"),
                "project_root": args.get("project_root"),
            },
        )

        # ---------------------------------------------------------------------
        # 3) Resolve engine adapter from input hints and snapshot required raw
        #    inputs into run-scoped storage for traceability/reproducibility.
        # ---------------------------------------------------------------------
        input_path = self._engine_detection_path(args)
        forced_engine = args.get("engine")
        self._console_step(args, f"Resolving engine input={input_path} forced_engine={forced_engine or 'auto'}")
        logger.debug("Resolving engine for input=%s forced_engine=%s", input_path, forced_engine)
        adapter = resolve_engine(input_path, engine=forced_engine)
        self._console_step(args, f"Resolved engine adapter={adapter.__class__.__name__}")
        logger.debug("Resolved adapter=%s", adapter.__class__.__name__)
        snapshot_names = adapter.required_input_files(required_data, args)
        required_source_names = tuple(snapshot_names or ())
        streaming = self._streaming_enabled(
            task,
            adapter,
            required_data,
            args,
            requested_frame_indices,
        )
        args["_streaming"] = streaming
        selective_sources: dict[str, str] = {}
        if requested_frame_indices is not None or streaming:
            snapshot_names, selective_sources = self._use_selective_sources(args, snapshot_names)
        self._console_step(args, "Snapshotting raw inputs")
        snapshot_storage_inputs(args, names=snapshot_names)
        if run_id := args.get("run_id"):
            project_root = Path(args.get("project_root") or ".")
            raw_dir = ReaxkitStorageLayout(project_root=project_root).raw_run_dir(str(run_id))
            copied = sorted([p.name for p in raw_dir.iterdir() if p.is_file()]) if raw_dir.exists() else []
            self._console_step(args, f"Snapshot ready raw_dir={raw_dir}")
            self._record_general(
                args,
                event="snapshot_raw_ready",
                task_name=task_name,
                extra={
                    "raw_dir": str(raw_dir),
                    "files": ",".join(copied),
                    "required_inputs": ",".join(snapshot_names or ()),
                    "selective_sources": ",".join(
                        f"{name}:{path}" for name, path in sorted(selective_sources.items())
                    ),
                    "engine_adapter": adapter.__class__.__name__,
                },
            )
        reporter = resolve_reporter(args)
        run_id = args.get("run_id")
        task_version = str(getattr(task, "VERSION", "1"))
        layout = ReaxkitStorageLayout(project_root=Path(args.get("project_root") or ".")) if run_id else None
        parsed_id = None

        # ---------------------------------------------------------------------
        # 4) If run-scoped layout is active, pre-register parsed dataset
        #    metadata (best effort) before loading/parsing input data.
        # ---------------------------------------------------------------------
        if layout is not None and requested_frame_indices is None and not streaming:
            try:
                handler_version = str(getattr(adapter, "HANDLER_VERSION", "1"))
                engine_name = adapter.__class__.__name__.replace("Adapter", "").lower()
                args["_parsed_id"] = layout.register_parsed_dataset(
                    run_id=str(run_id),
                    handler_version=handler_version,
                    engine=engine_name,
                )
                parsed_id = str(args["_parsed_id"])
                self._console_step(args, f"Registered parsed dataset parsed_id={parsed_id}")
                self._record_general(
                    args,
                    event="parsed_dataset_registered",
                    task_name=task_name,
                    extra={
                        "parsed_id": parsed_id,
                        "parsed_dir": str(layout.parsed_dir(parsed_id)),
                        "run_index": str(layout.run_index_path(str(run_id))),
                    },
                )
            except Exception as exc:  # pragma: no cover - best-effort metadata persistence
                logger.debug("Storage index update skipped: %s", exc)

        # ---------------------------------------------------------------------
        # 5) Prepare analysis cache and attempt fast path:
        #    - parsed artifact hit (typed data already materialized), then
        #    - analysis result cache hit/miss for the derived analysis_id.
        # ---------------------------------------------------------------------
        cache_root = Path(args.get("cache_dir") or (Path(input_path) / ".reaxkit_cache"))
        cache = CacheManager(CacheConfig(root=cache_root, namespace="analysis"))
        use_cache = bool(args.get("cache", True)) and not bool(args.get("no_cache", False))
        self._console_step(args, f"Analysis cache={'enabled' if use_cache else 'disabled'} root={cache_root}")
        analysis_id = None
        if streaming:
            identity = self._stream_source_identity(
                adapter,
                required_data,
                args,
                required_source_names,
            )
            analysis_id = cache.analysis_id_for(
                task=task,
                data=identity,
                request=request,
                task_version=task_version,
            )
            args["_analysis_id"] = analysis_id
            if use_cache and cache.exists(analysis_id):
                self._console_step(args, f"Analysis cache hit analysis_id={analysis_id[:12]} (returning cached result)")
                cached = cache.load(analysis_id)
                return enrich_result_with_time(
                    cached,
                    None,
                    control_file=str(args.get("control") or "control"),
                )

            self._console_step(args, f"Streaming data and running task={task_name}")
            t_stream0 = perf_counter()
            try:
                frames = adapter.stream(required_data, args, reporter=reporter)
                result = self._run_stream_task(task, frames, request, reporter)
            except (ParseError, AnalysisError):
                raise
            except Exception as exc:
                raise AnalysisError(
                    f"Task '{task_name}' failed during streaming analysis: {exc}"
                ) from exc
            result = enrich_result_with_time(
                result,
                None,
                control_file=str(args.get("control") or "control"),
            )
            elapsed = perf_counter() - t_stream0
            self._record_timing(
                args,
                phase="stream_analyze",
                task_name=task_name,
                seconds=elapsed,
            )
            if use_cache:
                cache.store(analysis_id, result, task_name=task_name)
                self._console_step(args, f"Stored analysis result in cache analysis_id={analysis_id[:12]}")
            self._record_general(
                args,
                event="analysis_done",
                task_name=task_name,
                extra={
                    "analysis_id": analysis_id,
                    "analysis_dir": str(self._analysis_output_dir(args)),
                    "status": "success",
                    "execution": "streaming",
                },
            )
            self._console_step(args, f"Completed streaming task={task_name} analysis_id={analysis_id[:12]}")
            return result
        if parsed_id is not None:
            data_name = getattr(required_data, "__name__", "parsed_data")
            artifact_name = str(data_name).lower()
            if layout is not None:
                self._console_step(args, f"Checking parsed artifact cache parsed_id={parsed_id} artifact={artifact_name}")
                cached_parsed = layout.load_parsed_artifact(
                    parsed_id=parsed_id,
                    artifact_name=artifact_name,
                )
                if cached_parsed is not None:
                    expected_type = required_data
                    if isinstance(cached_parsed, expected_type):
                        logger.debug("Parsed cache hit for data_type=%s parsed_id=%s", data_name, parsed_id[:12])
                        self._console_step(args, f"Parsed artifact cache hit parsed_id={parsed_id}")
                        t_load = 0.0
                        self._record_timing(args, phase="load_total", task_name=task_name, seconds=t_load)
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
                                task_name=task_name,
                                task_version=task_version,
                                user_settings=user_settings_from_args(args),
                            )
                        except Exception as exc:  # pragma: no cover - best-effort index update
                            logger.debug("Run analysis index update skipped: %s", exc)

                        if use_cache and cache.exists(analysis_id):
                            logger.info("Cache hit for task=%s analysis_id=%s", task_name, analysis_id[:12])
                            self._console_step(args, f"Analysis cache hit analysis_id={analysis_id[:12]} (returning cached result)")
                            self._record_general(
                                args,
                                event="analysis_cache_hit",
                                task_name=task_name,
                                extra={"analysis_id": analysis_id},
                            )
                            cached = cache.load(analysis_id)
                            return enrich_result_with_time(
                                cached,
                                data,
                                control_file=str(args.get("control") or "control"),
                            )

                        if not use_cache:
                            logger.info("Cache disabled; executing task=%s", task_name)
                            self._console_step(args, "Analysis cache disabled (running task)")
                        else:
                            logger.info("Cache miss for task=%s analysis_id=%s", task_name, analysis_id[:12])
                            self._console_step(args, f"Analysis cache miss analysis_id={analysis_id[:12]} (running task)")
                            self._record_general(
                                args,
                                event="analysis_cache_miss",
                                task_name=task_name,
                                extra={"analysis_id": analysis_id},
                            )
                        self._console_step(args, f"Running analysis task={task_name}")
                        t_run0 = perf_counter()
                        try:
                            result = self._run_task(task, data, request, reporter)
                        except AnalysisError:
                            raise
                        except Exception as exc:
                            raise AnalysisError(
                                f"Task '{task_name}' failed during analysis: {exc}"
                            ) from exc
                        result = enrich_result_with_time(
                            result,
                            data,
                            control_file=str(args.get("control") or "control"),
                        )
                        t_run = perf_counter() - t_run0
                        self._record_timing(args, phase="analyze", task_name=task_name, seconds=t_run)
                        if use_cache:
                            cache.store(analysis_id, result, task_name=task_name)
                            logger.debug("Stored result in cache analysis_id=%s", analysis_id[:12])
                            self._console_step(args, f"Stored analysis result in cache analysis_id={analysis_id[:12]}")
                        self._record_general(
                            args,
                            event="analysis_done",
                            task_name=task_name,
                            extra={
                                "analysis_id": analysis_id,
                                "analysis_dir": str(self._analysis_output_dir(args)),
                                "status": "success",
                            },
                        )
                        self._console_step(args, f"Completed task={task_name} analysis_id={analysis_id[:12]}")
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
                    task_name=task_name,
                    task_version=task_version,
                    user_settings=user_settings_from_args(args),
                )
            except Exception as exc:  # pragma: no cover - best-effort index update
                logger.debug("Run analysis index update skipped: %s", exc)

            if use_cache and cache.exists(analysis_id):
                logger.info("Cache hit for task=%s analysis_id=%s", task_name, analysis_id[:12])
                self._console_step(args, f"Analysis cache hit analysis_id={analysis_id[:12]} (returning cached result)")
                self._record_general(
                    args,
                    event="analysis_cache_hit",
                    task_name=task_name,
                    extra={"analysis_id": analysis_id},
                )
                cached = cache.load(analysis_id)
                return enrich_result_with_time(
                cached,
                None,
                control_file=str(args.get("control") or "control"),
            )

        # ---------------------------------------------------------------------
        # 6) Slow path data load: parse required typed data via adapter.
        # ---------------------------------------------------------------------
        t_load0 = perf_counter()
        self._console_step(args, f"Loading data via adapter={adapter.__class__.__name__} data_type={data_name}")
        try:
            data = adapter.load(required_data, args, reporter=reporter)
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError(
                f"Failed to load required data '{getattr(required_data, '__name__', str(required_data))}' "
                f"for task '{task_name}': {exc}"
            ) from exc
        self._console_step(args, "Data loading complete")
        if run_id and layout is not None and parsed_id is None and requested_frame_indices is None:
            try:
                handler_version = str(getattr(adapter, "HANDLER_VERSION", "1"))
                engine_name = adapter.__class__.__name__.replace("Adapter", "").lower()
                args["_parsed_id"] = layout.register_parsed_dataset(
                    run_id=str(run_id),
                    handler_version=handler_version,
                    engine=engine_name,
                )
                parsed_id = str(args["_parsed_id"])
                self._console_step(args, f"Registered parsed dataset parsed_id={parsed_id}")
                self._record_general(
                    args,
                    event="parsed_dataset_registered",
                    task_name=task_name,
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
                data_name = getattr(required_data, "__name__", "parsed_data")
                artifact_name = data_name.lower()
                parsed_artifact_path = layout.persist_parsed_artifact(
                    parsed_id=parsed_id,
                    artifact_name=artifact_name,
                    data=data,
                )
                self._record_general(
                    args,
                    event="parsed_artifact_saved",
                    task_name=task_name,
                    extra={
                        "parsed_id": parsed_id,
                        "artifact": artifact_name,
                        "path": str(parsed_artifact_path),
                    },
                )
                self._console_step(args, f"Saved parsed artifact path={parsed_artifact_path}")
            except Exception as exc:  # pragma: no cover - best-effort artifact persistence
                logger.debug("Parsed artifact write skipped: %s", exc)
        t_load = perf_counter() - t_load0
        logger.debug(
            "Loaded data_type=%s for task=%s",
            getattr(required_data, "__name__", str(required_data)),
            task_name,
        )
        self._record_timing(args, phase="load_total", task_name=task_name, seconds=t_load)

        # ---------------------------------------------------------------------
        # 7) Compute analysis_id (if needed), update run index metadata, then
        #    execute cache decision (disabled / hit / miss).
        # ---------------------------------------------------------------------
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
                    task_name=task_name,
                    task_version=task_version,
                    user_settings=user_settings_from_args(args),
                )
            except Exception as exc:  # pragma: no cover - best-effort index update
                logger.debug("Run analysis index update skipped: %s", exc)

        # ---------------------------------------------------------------------
        # 8) Execute analysis task, enrich result with time axis metadata,
        #    persist cache/log metadata, and return final result object.
        # ---------------------------------------------------------------------
        if not use_cache:
            logger.info("Cache disabled; executing task=%s", task_name)
            self._console_step(args, "Analysis cache disabled (running task)")
            self._console_step(args, f"Running analysis task={task_name}")
            t_run0 = perf_counter()
            try:
                result = self._run_task(task, data, request, reporter)
            except AnalysisError:
                raise
            except Exception as exc:
                raise AnalysisError(
                    f"Task '{task_name}' failed during analysis: {exc}"
                ) from exc
            result = enrich_result_with_time(
                result,
                data,
                control_file=str(args.get("control") or "control"),
            )
            t_run = perf_counter() - t_run0
            self._record_timing(args, phase="analyze", task_name=task_name, seconds=t_run)
            self._record_general(
                args,
                event="analysis_done",
                task_name=task_name,
                extra={
                    "analysis_id": analysis_id,
                    "analysis_dir": str(self._analysis_output_dir(args)),
                    "status": "success",
                },
            )
            self._console_step(args, f"Completed task={task_name} analysis_id={analysis_id[:12]}")
            return result

        if cache.exists(analysis_id):
            logger.info("Cache hit for task=%s analysis_id=%s", task_name, analysis_id[:12])
            self._console_step(args, f"Analysis cache hit analysis_id={analysis_id[:12]} (returning cached result)")
            self._record_general(
                args,
                event="analysis_cache_hit",
                task_name=task_name,
                extra={"analysis_id": analysis_id},
            )
            cached = cache.load(analysis_id)
            return enrich_result_with_time(
                cached,
                data,
                control_file=str(args.get("control") or "control"),
            )

        logger.info("Cache miss for task=%s analysis_id=%s", task_name, analysis_id[:12])
        self._console_step(args, f"Analysis cache miss analysis_id={analysis_id[:12]} (running task)")
        self._record_general(
            args,
            event="analysis_cache_miss",
            task_name=task_name,
            extra={"analysis_id": analysis_id},
        )
        self._console_step(args, f"Running analysis task={task_name}")
        t_run0 = perf_counter()
        try:
            result = self._run_task(task, data, request, reporter)
        except AnalysisError:
            raise
        except Exception as exc:
            raise AnalysisError(
                f"Task '{task_name}' failed during analysis: {exc}"
            ) from exc
        result = enrich_result_with_time(
            result,
            data,
            control_file=str(args.get("control") or "control"),
        )
        t_run = perf_counter() - t_run0
        self._record_timing(args, phase="analyze", task_name=task_name, seconds=t_run)
        cache.store(analysis_id, result, task_name=task_name)
        self._console_step(args, f"Stored analysis result in cache analysis_id={analysis_id[:12]}")
        logger.debug("Stored result in cache analysis_id=%s", analysis_id[:12])
        self._record_general(
            args,
            event="analysis_done",
            task_name=task_name,
            extra={
                "analysis_id": analysis_id,
                "analysis_dir": str(self._analysis_output_dir(args)),
                "status": "success",
            },
        )
        self._console_step(args, f"Completed task={task_name} analysis_id={analysis_id[:12]}")
        return result
