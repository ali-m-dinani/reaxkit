"""Shared contracts and automatic policy selection for frame pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import os
from typing import Any, Mapping, Protocol, runtime_checkable


class ExecutionShape(str, Enum):
    """Frame dependency shape declared by an analysis task."""

    INDEPENDENT_FRAME_MAP = "independent_frame_map"
    REFERENCE_FRAME_MAP = "reference_frame_map"
    STREAMING_REDUCTION = "streaming_reduction"
    ORDERED_STATEFUL_STREAM = "ordered_stateful_stream"
    GLOBAL = "global"


@dataclass(frozen=True, slots=True)
class TaskCapabilities:
    """Execution properties used by the shared runtime policy."""

    shape: ExecutionShape = ExecutionShape.GLOBAL
    thread_safe: bool = False
    deterministic_order: bool = True
    needs_reference: bool = False
    estimated_frame_bytes: int | None = None
    estimated_result_bytes: int | None = None
    max_workers: int | None = None
    required_fields: tuple[str, ...] = ()
    data_sources: tuple[str, ...] = ()

    @property
    def supports_frame_parallelism(self) -> bool:
        return bool(
            self.thread_safe
            and self.shape
            in {
                ExecutionShape.INDEPENDENT_FRAME_MAP,
                ExecutionShape.REFERENCE_FRAME_MAP,
                ExecutionShape.STREAMING_REDUCTION,
            }
        )


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """Resolved worker, queue, and memory limits for one analysis run."""

    workers: int
    max_in_flight: int
    backend: str
    allocated_cpus: int
    memory_limit_bytes: int | None
    estimated_frame_bytes: int | None
    worker_source: str
    queue_source: str
    execution_shape: str = ExecutionShape.GLOBAL.value
    decision_reason: str = "conservative serial fallback"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FrameEnvelope:
    """One bounded unit exchanged between a reader and a frame kernel."""

    sequence: int
    source_frame: int
    payload: Any
    estimated_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FrameResult:
    """A compact frame result paired with its input envelope."""

    envelope: FrameEnvelope
    value: Any


@dataclass(frozen=True, slots=True)
class PreparedState:
    """Immutable command state built once and shared by frame kernels."""

    payload: Any
    estimated_bytes: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class FrameProducer(Protocol):
    def __iter__(self): ...


@runtime_checkable
class InputAligner(Protocol):
    def __call__(self, value: Any, sequence: int) -> FrameEnvelope: ...


@runtime_checkable
class FrameKernel(Protocol):
    def __call__(self, payload: Any) -> Any: ...


@runtime_checkable
class FrameReducer(Protocol):
    def add(self, result: FrameResult) -> None: ...

    def finish(self) -> Any: ...


@runtime_checkable
class ArtifactSink(Protocol):
    def append(self, result: FrameResult) -> None: ...

    def finalize(self) -> Any: ...


def task_capabilities(task: Any) -> TaskCapabilities:
    """Return an explicit declaration or a conservative inferred contract."""
    declared = getattr(task, "execution_capabilities", None)
    if isinstance(declared, TaskCapabilities):
        return declared
    if isinstance(declared, Mapping):
        values = dict(declared)
        if "shape" in values and not isinstance(values["shape"], ExecutionShape):
            values["shape"] = ExecutionShape(str(values["shape"]))
        return TaskCapabilities(**values)
    from reaxkit.core.runtime.priority_task_manifest import classification_for_task

    classification = classification_for_task(task)
    if classification is not None:
        return TaskCapabilities(
            shape=ExecutionShape(str(classification["shape"])),
            thread_safe=bool(classification["thread_safe"]),
        )
    from reaxkit.core.runtime.analysis_task_manifest import (
        classification_for_registered_task,
    )

    classification = classification_for_registered_task(task)
    if classification is not None:
        return TaskCapabilities(
            shape=ExecutionShape(str(classification["shape"])),
            thread_safe=bool(classification["thread_safe"]),
        )
    if callable(getattr(task, "run_stream", None)):
        return TaskCapabilities(shape=ExecutionShape.ORDERED_STATEFUL_STREAM)
    return TaskCapabilities()


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _allocated_cpus(args: Mapping[str, Any], environ: Mapping[str, str]) -> int:
    for value in (
        args.get("allocated_cpus"),
        args.get("cpus_per_task"),
        environ.get("SLURM_CPUS_PER_TASK"),
    ):
        if parsed := _positive_int(value):
            return parsed
    return max(1, int(os.cpu_count() or 1))


def _memory_limit_bytes(args: Mapping[str, Any], environ: Mapping[str, str], cpus: int) -> int | None:
    if parsed := _positive_int(args.get("memory_limit_bytes")):
        return parsed
    if parsed := _positive_int(environ.get("SLURM_MEM_PER_NODE")):
        return parsed * 1024 * 1024
    if parsed := _positive_int(environ.get("SLURM_MEM_PER_CPU")):
        return parsed * 1024 * 1024 * cpus
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except (ImportError, OSError):
        try:
            pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return pages * page_size
        except (AttributeError, OSError, TypeError, ValueError):
            return None


def resolve_execution_policy(
    task: Any,
    request: Any,
    args: Mapping[str, Any] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ExecutionPolicy:
    """Choose safe defaults while honoring positive expert overrides."""
    settings = args or {}
    env = os.environ if environ is None else environ
    capabilities = task_capabilities(task)
    cpus = _allocated_cpus(settings, env)
    memory_bytes = _memory_limit_bytes(settings, env, cpus)

    requested_workers = _positive_int(
        getattr(request, "workers", None) or settings.get("workers")
    )
    can_parallelize = capabilities.supports_frame_parallelism
    if not can_parallelize:
        workers = 1
        worker_source = "capability_serial"
        decision_reason = (
            "ordered state must be preserved"
            if capabilities.shape is ExecutionShape.ORDERED_STATEFUL_STREAM
            else "global algorithm requires a specialized blocked implementation"
            if capabilities.shape is ExecutionShape.GLOBAL
            else "task has no thread-safe shared frame kernel"
        )
    elif requested_workers is not None:
        workers = min(requested_workers, cpus)
        worker_source = "explicit"
        decision_reason = "explicit worker override within allocation"
    else:
        workers = cpus
        worker_source = "automatic"
        decision_reason = "thread-safe frame kernel uses allocated CPUs"
    if capabilities.max_workers is not None:
        workers = min(workers, max(1, int(capabilities.max_workers)))
    workers = max(1, workers)

    requested_queue = _positive_int(
        getattr(request, "chunk_size", None) or settings.get("chunk_size")
    )
    if requested_queue is not None:
        max_in_flight = requested_queue
        workers = min(workers, max_in_flight)
        queue_source = "explicit"
    else:
        max_in_flight = max(1, workers * 2)
        queue_source = "automatic"

    estimate = capabilities.estimated_frame_bytes
    result_estimate = capabilities.estimated_result_bytes or 0
    work_item_estimate = (estimate or 0) + result_estimate
    if memory_bytes and work_item_estimate > 0:
        # At most one quarter of the allocation is assigned to queued frame
        # payloads. The remainder stays available to the parser, reference
        # state, workers, reducers, plotting, and Python itself.
        memory_items = max(1, int((memory_bytes * 0.25) // work_item_estimate))
        if memory_items < max_in_flight:
            max_in_flight = memory_items
            queue_source = f"{queue_source}+memory_cap"
        workers = min(workers, max_in_flight)

    backend = (
        "blocked_serial"
        if capabilities.shape is ExecutionShape.GLOBAL
        and callable(getattr(task, "run_blocks", None))
        else "thread"
        if workers > 1
        else "serial"
    )
    return ExecutionPolicy(
        workers=workers,
        max_in_flight=max(1, max_in_flight),
        backend=backend,
        allocated_cpus=cpus,
        memory_limit_bytes=memory_bytes,
        estimated_frame_bytes=estimate,
        worker_source=worker_source,
        queue_source=queue_source,
        execution_shape=capabilities.shape.value,
        decision_reason=decision_reason,
    )


__all__ = [
    "ArtifactSink",
    "ExecutionPolicy",
    "ExecutionShape",
    "FrameEnvelope",
    "FrameKernel",
    "FrameProducer",
    "FrameReducer",
    "FrameResult",
    "InputAligner",
    "PreparedState",
    "TaskCapabilities",
    "resolve_execution_policy",
    "task_capabilities",
]
