"""Bounded, ordered frame scheduling shared by streaming analyses."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext, closing
from dataclasses import dataclass, fields, is_dataclass, replace
from functools import partial
from threading import Lock
from time import perf_counter
from typing import Any, Callable, Iterable, Iterator

import numpy as np
from threadpoolctl import threadpool_limits

from reaxkit.core.runtime.execution_contracts import (
    ExecutionPolicy,
    FrameEnvelope,
    FrameResult,
    PreparedState,
)


@dataclass(slots=True)
class PipelineMetrics:
    frames_read: int = 0
    frames_submitted: int = 0
    frames_completed: int = 0
    reader_seconds: float = 0.0
    worker_seconds: float = 0.0
    collector_wait_seconds: float = 0.0
    total_seconds: float = 0.0
    peak_in_flight: int = 0
    peak_in_flight_bytes: int = 0
    cancelled_frames: int = 0

    def common_details(self, policy: ExecutionPolicy) -> dict[str, Any]:
        return {
            "frames_read": self.frames_read,
            "frames_submitted": self.frames_submitted,
            "frames_completed": self.frames_completed,
            "peak_in_flight": self.peak_in_flight,
            "peak_in_flight_bytes": self.peak_in_flight_bytes,
            "workers": policy.workers,
            "max_in_flight": policy.max_in_flight,
            "backend": policy.backend,
        }


TimingCallback = Callable[[str, float, dict[str, Any]], None]


def estimate_payload_bytes(value: Any, *, _seen: set[int] | None = None) -> int:
    """Estimate retained array/buffer bytes without traversing arbitrary objects."""
    if value is None:
        return 0
    seen = set() if _seen is None else _seen
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return int(len(value))
    if isinstance(value, dict):
        return sum(estimate_payload_bytes(item, _seen=seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(estimate_payload_bytes(item, _seen=seen) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return sum(
            estimate_payload_bytes(getattr(value, descriptor.name), _seen=seen)
            for descriptor in fields(value)
        )
    return 0


class BoundedFramePipeline:
    """Overlap one forward reader with ordered, bounded frame calculation."""

    def __init__(
        self,
        policy: ExecutionPolicy,
        *,
        timing_callback: TimingCallback | None = None,
        artifact_writer: Any | None = None,
    ) -> None:
        self.policy = policy
        self.metrics = PipelineMetrics()
        self._timing_callback = timing_callback
        self.artifact_writer = artifact_writer
        self._used = False
        self._worker_lock = Lock()

    @contextmanager
    def measure_stage(self, phase: str, **details: Any):
        started = perf_counter()
        try:
            yield
        finally:
            if self._timing_callback is not None:
                self._timing_callback(phase, perf_counter() - started, details)

    @staticmethod
    def _source_frame(payload: Any, sequence: int) -> int:
        for name in ("source_frame", "frame_index", "frame_idx"):
            value = getattr(payload, name, None)
            if value is not None:
                return int(value)
        for candidate in (payload, getattr(payload, "trajectory", None), getattr(payload, "connectivity", None)):
            indices = getattr(candidate, "source_frame_indices", None)
            if indices is not None and len(indices):
                return int(indices[0])
        return int(sequence)

    def _envelope(self, payload: Any, sequence: int) -> FrameEnvelope:
        if isinstance(payload, FrameEnvelope):
            return payload if payload.estimated_bytes > 0 else replace(
                payload, estimated_bytes=estimate_payload_bytes(payload.payload)
            )
        estimated = estimate_payload_bytes(payload)
        if estimated <= 0 and self.policy.estimated_frame_bytes:
            estimated = int(self.policy.estimated_frame_bytes)
        return FrameEnvelope(
            sequence=sequence,
            source_frame=self._source_frame(payload, sequence),
            payload=payload,
            estimated_bytes=estimated,
        )

    def _work(self, kernel: Callable[[Any], Any], envelope: FrameEnvelope) -> Any:
        started = perf_counter()
        try:
            return kernel(envelope.payload)
        except Exception as exc:
            raise RuntimeError(f"Frame kernel failed at source frame {envelope.source_frame}: {exc}") from exc
        finally:
            elapsed = perf_counter() - started
            with self._worker_lock:
                self.metrics.worker_seconds += elapsed

    def _emit_metrics(self) -> None:
        if self._timing_callback is None:
            return
        details = self.metrics.common_details(self.policy)
        self._timing_callback("pipeline_reader", self.metrics.reader_seconds, details)
        self._timing_callback("pipeline_workers", self.metrics.worker_seconds, details)
        self._timing_callback(
            "pipeline_collector_wait", self.metrics.collector_wait_seconds, details
        )
        self._timing_callback("pipeline_total", self.metrics.total_seconds, details)

    def map_ordered(
        self,
        source: Iterable[Any],
        kernel: Callable[[Any], Any],
    ) -> Iterator[FrameResult]:
        """Yield compact results in input order with bounded read-ahead."""
        budget = threadpool_limits(limits=1) if self.policy.workers > 1 else nullcontext()
        with budget, closing(self._map_ordered(source, kernel)) as results:
            yield from results

    def _map_ordered(self, source, kernel) -> Iterator[FrameResult]:
        if self._used:
            raise RuntimeError("A BoundedFramePipeline instance can only be consumed once.")
        self._used = True
        started = perf_counter()
        iterator = iter(source)

        if self.policy.workers == 1:
            try:
                sequence = 0
                while True:
                    read_started = perf_counter()
                    try:
                        payload = next(iterator)
                    except StopIteration:
                        self.metrics.reader_seconds += perf_counter() - read_started
                        break
                    self.metrics.reader_seconds += perf_counter() - read_started
                    envelope = self._envelope(payload, sequence)
                    sequence += 1
                    self.metrics.frames_read += 1
                    self.metrics.frames_submitted += 1
                    self.metrics.peak_in_flight = max(self.metrics.peak_in_flight, 1)
                    self.metrics.peak_in_flight_bytes = max(
                        self.metrics.peak_in_flight_bytes, envelope.estimated_bytes
                    )
                    value = self._work(kernel, envelope)
                    self.metrics.frames_completed += 1
                    yield FrameResult(envelope=envelope, value=value)
            finally:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
                self.metrics.total_seconds = perf_counter() - started
                self._emit_metrics()
            return

        pending: deque[tuple[FrameEnvelope, Future[Any]]] = deque()
        pending_bytes = 0
        exhausted = False
        pool = ThreadPoolExecutor(
            max_workers=self.policy.workers,
            thread_name_prefix="reaxkit-frame",
        )
        try:
            sequence = 0
            while pending or not exhausted:
                while not exhausted and len(pending) < self.policy.max_in_flight:
                    # Notice a later worker's failure before advancing a potentially
                    # expensive reader, even when an earlier frame is still running.
                    for _, submitted in pending:
                        if submitted.done() and submitted.exception() is not None:
                            submitted.result()
                    read_started = perf_counter()
                    try:
                        payload = next(iterator)
                    except StopIteration:
                        self.metrics.reader_seconds += perf_counter() - read_started
                        exhausted = True
                        break
                    self.metrics.reader_seconds += perf_counter() - read_started
                    envelope = self._envelope(payload, sequence)
                    sequence += 1
                    future = pool.submit(self._work, kernel, envelope)
                    pending.append((envelope, future))
                    pending_bytes += envelope.estimated_bytes
                    self.metrics.frames_read += 1
                    self.metrics.frames_submitted += 1
                    self.metrics.peak_in_flight = max(
                        self.metrics.peak_in_flight, len(pending)
                    )
                    self.metrics.peak_in_flight_bytes = max(
                        self.metrics.peak_in_flight_bytes, pending_bytes
                    )

                if not pending:
                    continue
                envelope, future = pending.popleft()
                wait_started = perf_counter()
                value = future.result()
                self.metrics.collector_wait_seconds += perf_counter() - wait_started
                pending_bytes -= envelope.estimated_bytes
                self.metrics.frames_completed += 1
                yield FrameResult(envelope=envelope, value=value)
        except BaseException:
            for _envelope, future in pending:
                if future.cancel():
                    self.metrics.cancelled_frames += 1
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
            self.metrics.total_seconds = perf_counter() - started
            self._emit_metrics()

    def map_reference(self, source, prepare, kernel) -> Iterator[FrameResult]:
        """Prepare shared state once, then run ``kernel(payload, state)`` in order.

        The caller supplies dependency frames to ``prepare`` separately from the
        selected source, so dependency-only frames cannot leak into the result.
        """
        with self.measure_stage("reference_preparation"):
            state = prepare()
            if not isinstance(state, PreparedState):
                state = PreparedState(state)
            _readonly_arrays(state.payload)
        yield from self.map_ordered(source, partial(kernel, state=state.payload))

    def scan_ordered(self, source, state, step) -> Iterator[FrameResult]:
        """Apply ``step(state, payload) -> (state, result)`` strictly serially."""
        if self.policy.workers != 1:
            raise ValueError("Ordered scans require a serial execution policy.")

        def advance(payload):
            nonlocal state
            state, result = step(state, payload)
            return result

        yield from self.map_ordered(source, advance)

    def reduce_ordered(self, source, kernel, reducer):
        """Merge frame contributions in deterministic order into bounded state."""
        with closing(self.map_ordered(source, kernel)) as results:
            for result in results:
                reducer.add(result.value)
        return reducer.finalize()

    def iter_blocks(
        self,
        source: Iterable[Any],
        *,
        block_size: int | None = None,
    ) -> Iterator[tuple[FrameEnvelope, ...]]:
        """Yield bounded input blocks for algorithms that need frame windows.

        This path deliberately performs no generic parallel calculation. A
        global task owns the mathematics within and between blocks and can
        merge block state without materializing the complete trajectory.
        """
        if self._used:
            raise RuntimeError("A BoundedFramePipeline instance can only be consumed once.")
        self._used = True
        limit = max(1, int(block_size or self.policy.max_in_flight))
        iterator = iter(source)
        started = perf_counter()
        sequence = 0
        try:
            while True:
                block: list[FrameEnvelope] = []
                block_bytes = 0
                while len(block) < limit:
                    read_started = perf_counter()
                    try:
                        payload = next(iterator)
                    except StopIteration:
                        self.metrics.reader_seconds += perf_counter() - read_started
                        break
                    self.metrics.reader_seconds += perf_counter() - read_started
                    envelope = self._envelope(payload, sequence)
                    sequence += 1
                    block.append(envelope)
                    block_bytes += envelope.estimated_bytes
                    self.metrics.frames_read += 1
                    self.metrics.frames_submitted += 1
                if not block:
                    break
                self.metrics.peak_in_flight = max(
                    self.metrics.peak_in_flight, len(block)
                )
                self.metrics.peak_in_flight_bytes = max(
                    self.metrics.peak_in_flight_bytes, block_bytes
                )
                yield tuple(block)
                self.metrics.frames_completed += len(block)
                if len(block) < limit:
                    break
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
            self.metrics.total_seconds = perf_counter() - started
            self._emit_metrics()


def _readonly_arrays(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    elif isinstance(value, dict):
        for item in value.values():
            _readonly_arrays(item, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _readonly_arrays(item, seen)
    elif is_dataclass(value) and not isinstance(value, type):
        for descriptor in fields(value):
            _readonly_arrays(getattr(value, descriptor.name), seen)


__all__ = ["BoundedFramePipeline", "PipelineMetrics", "estimate_payload_bytes"]
