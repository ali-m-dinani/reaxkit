"""
Core helpers for runtime.

This module provides core building blocks that are shared across runtime, storage, registry, and study orchestration flows in ReaxKit.

**Usage context**

- Import these helpers from ReaxKit core modules when implementing CLI and workflow logic.
- Reuse the public APIs here to keep behavior consistent across commands and engines.
"""

from reaxkit.core.runtime.execution_contracts import (
    ArtifactSink,
    ExecutionPolicy,
    ExecutionShape,
    FrameEnvelope,
    FrameKernel,
    FrameProducer,
    FrameReducer,
    FrameResult,
    InputAligner,
    PreparedState,
    TaskCapabilities,
    resolve_execution_policy,
    task_capabilities,
)
from reaxkit.core.runtime.frame_pipeline import (
    BoundedFramePipeline,
    PipelineMetrics,
    estimate_payload_bytes,
)
from reaxkit.core.runtime.artifacts import (
    ArtifactSpec,
    ArtifactWriter,
    BufferedTableSink,
    TableChunks,
)
from reaxkit.core.runtime.reducers import (
    CountSumReducer,
    HistogramReducer,
    PlotMatrixReducer,
    TableAccumulator,
)
from reaxkit.core.runtime.analysis_task_manifest import (
    ALL_GENERAL_TASKS,
    GLOBAL_TASKS,
    SHARED_PIPELINE_TASKS,
)

__all__ = [
    "ArtifactSink",
    "ArtifactSpec",
    "ArtifactWriter",
    "ALL_GENERAL_TASKS",
    "BoundedFramePipeline",
    "BufferedTableSink",
    "CountSumReducer",
    "ExecutionPolicy",
    "ExecutionShape",
    "FrameEnvelope",
    "FrameKernel",
    "FrameProducer",
    "FrameReducer",
    "FrameResult",
    "GLOBAL_TASKS",
    "InputAligner",
    "HistogramReducer",
    "PipelineMetrics",
    "PlotMatrixReducer",
    "PreparedState",
    "SHARED_PIPELINE_TASKS",
    "TaskCapabilities",
    "TableAccumulator",
    "TableChunks",
    "estimate_payload_bytes",
    "resolve_execution_policy",
    "task_capabilities",
]
