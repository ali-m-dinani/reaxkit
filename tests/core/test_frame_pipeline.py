from __future__ import annotations

from types import SimpleNamespace
import time

import pytest

from reaxkit.core.runtime.execution_contracts import (
    ExecutionPolicy,
    ExecutionShape,
    TaskCapabilities,
    resolve_execution_policy,
    task_capabilities,
)
from reaxkit.core.runtime.frame_pipeline import BoundedFramePipeline
from reaxkit.core.runtime.priority_task_manifest import PRIORITY_TASK_MANIFEST
from reaxkit.core.runtime.analysis_task_manifest import ALL_GENERAL_TASKS


class _ParallelTask:
    execution_capabilities = TaskCapabilities(
        shape=ExecutionShape.INDEPENDENT_FRAME_MAP,
        thread_safe=True,
        estimated_frame_bytes=100,
    )


class _OrderedTask:
    execution_capabilities = TaskCapabilities(
        shape=ExecutionShape.ORDERED_STATEFUL_STREAM,
        thread_safe=False,
    )


def test_automatic_policy_uses_slurm_cpu_allocation_and_bounded_read_ahead():
    policy = resolve_execution_policy(
        _ParallelTask(),
        SimpleNamespace(workers=0, chunk_size=0),
        {},
        environ={"SLURM_CPUS_PER_TASK": "4", "SLURM_MEM_PER_NODE": "1024"},
    )

    assert policy.workers == 4
    assert policy.max_in_flight == 8
    assert policy.worker_source == "automatic"
    assert policy.queue_source == "automatic"


def test_policy_honors_overrides_but_caps_queue_by_memory():
    policy = resolve_execution_policy(
        _ParallelTask(),
        SimpleNamespace(workers=3, chunk_size=12),
        {"memory_limit_bytes": 1000},
        environ={"SLURM_CPUS_PER_TASK": "8"},
    )

    assert policy.workers == 2
    assert policy.max_in_flight == 2
    assert policy.worker_source == "explicit"
    assert policy.queue_source == "explicit+memory_cap"


def test_stateful_capability_forces_serial_execution():
    policy = resolve_execution_policy(
        _OrderedTask(),
        SimpleNamespace(workers=8, chunk_size=16),
        {},
        environ={"SLURM_CPUS_PER_TASK": "8"},
    )

    assert policy.workers == 1
    assert policy.backend == "serial"
    assert policy.worker_source == "capability_serial"


def test_bounded_pipeline_preserves_order_and_reports_queue_metrics():
    records: list[tuple[str, float, dict]] = []
    policy = ExecutionPolicy(
        workers=3,
        max_in_flight=4,
        backend="thread",
        allocated_cpus=3,
        memory_limit_bytes=None,
        estimated_frame_bytes=10,
        worker_source="test",
        queue_source="test",
    )
    pipeline = BoundedFramePipeline(
        policy,
        timing_callback=lambda phase, seconds, details: records.append(
            (phase, seconds, details)
        ),
    )

    def kernel(value: int) -> int:
        time.sleep(0.002 * (6 - value))
        return value * 10

    completed = list(pipeline.map_ordered(range(6), kernel))

    assert [item.envelope.source_frame for item in completed] == list(range(6))
    assert [item.value for item in completed] == [0, 10, 20, 30, 40, 50]
    assert pipeline.metrics.peak_in_flight <= 4
    assert pipeline.metrics.frames_read == 6
    assert pipeline.metrics.frames_completed == 6
    assert {phase for phase, _, _ in records} == {
        "pipeline_reader",
        "pipeline_workers",
        "pipeline_collector_wait",
        "pipeline_total",
    }


def test_bounded_pipeline_cancels_pending_work_after_worker_failure():
    policy = ExecutionPolicy(
        workers=2,
        max_in_flight=3,
        backend="thread",
        allocated_cpus=2,
        memory_limit_bytes=None,
        estimated_frame_bytes=None,
        worker_source="test",
        queue_source="test",
    )
    pipeline = BoundedFramePipeline(policy)
    source_closed = []

    def source():
        try:
            yield from range(10)
        finally:
            source_closed.append(True)

    def kernel(value: int) -> int:
        if value == 1:
            raise RuntimeError("broken frame")
        time.sleep(0.01)
        return value

    with pytest.raises(RuntimeError, match="broken frame"):
        list(pipeline.map_ordered(source(), kernel))

    assert source_closed == [True]
    assert pipeline.metrics.frames_submitted <= 4
    assert pipeline.metrics.frames_completed < pipeline.metrics.frames_submitted


def test_priority_manifest_classifies_all_electrostatics_and_ferroelectrics_commands():
    commands = [record["command"] for record in PRIORITY_TASK_MANIFEST]

    assert len(commands) == 25
    assert len(set(commands)) == len(commands)
    assert {record["shape"] for record in PRIORITY_TASK_MANIFEST} == {
        shape.value for shape in ExecutionShape if shape is not ExecutionShape.SINGLE
    }
    assert all(record["status"].startswith(("migrated_", "classified_")) for record in PRIORITY_TASK_MANIFEST)


def test_explicit_capabilities_take_precedence_over_manifest_classification():
    from reaxkit.analysis.electrostatics.electrostatics import DipoleTask

    capabilities = task_capabilities(DipoleTask())

    assert capabilities.shape is ExecutionShape.INDEPENDENT_FRAME_MAP
    assert capabilities.thread_safe
    assert capabilities.estimated_frame_bytes == 8 * 1024 * 1024


def test_general_manifest_covers_every_automatically_registered_analysis_task():
    import reaxkit.analysis  # noqa: F401
    from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY

    priority_commands = {record["command"] for record in PRIORITY_TASK_MANIFEST}
    assert set(TASK_REGISTRY) <= set(ALL_GENERAL_TASKS) | priority_commands


def test_global_block_iterator_is_bounded_and_preserves_order():
    policy = ExecutionPolicy(
        workers=1,
        max_in_flight=3,
        backend="blocked_serial",
        allocated_cpus=1,
        memory_limit_bytes=None,
        estimated_frame_bytes=8,
        worker_source="test",
        queue_source="test",
        execution_shape="global",
        decision_reason="test",
    )
    pipeline = BoundedFramePipeline(policy)

    blocks = list(pipeline.iter_blocks(range(8)))

    assert [[item.payload for item in block] for block in blocks] == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7],
    ]
    assert pipeline.metrics.frames_completed == 8
    assert pipeline.metrics.peak_in_flight == 3
    assert pipeline.metrics.peak_in_flight_bytes <= 24
