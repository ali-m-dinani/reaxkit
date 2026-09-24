"""Explicit adapter for independent, single-table frame kernels."""

from contextlib import closing
from dataclasses import replace

import pandas as pd

from reaxkit.core.runtime.execution_contracts import FrameEnvelope, resolve_execution_policy
from reaxkit.core.runtime.frame_pipeline import BoundedFramePipeline, estimate_payload_bytes


def source_frame_index(data, fallback):
    for candidate in (data, getattr(data, "trajectory", None), getattr(data, "connectivity", None), getattr(data, "simulation", None)):
        indices = getattr(candidate, "source_frame_indices", None)
        if indices is None:
            indices = (getattr(candidate, "metadata", None) or {}).get("source_frame_indices")
        if indices is not None and len(indices):
            return int(indices[0])
    return fallback


def selected_frame_envelopes(frames, request):
    stride = max(1, int(request.every))
    selection = getattr(request, "selected_frames", None)
    if selection is None:
        selection = getattr(request, "frames", None)
    wanted = None if selection is None else set(list(selection)[::stride])
    iterator = iter(frames)
    position = 0
    try:
        for fallback, data in enumerate(iterator):
            source = source_frame_index(data, fallback)
            if wanted is not None and source not in wanted:
                continue
            take = wanted is not None or position % stride == 0
            position += 1
            if take:
                yield FrameEnvelope(position - 1, source, data, estimate_payload_bytes(data))
    finally:
        close = getattr(iterator, "close", None)
        if close:
            close()


def map_frame_tables(task, frames, request, *, pipeline=None, reporter=None, sort_columns=()):
    """Run an audited task's serial kernel on single-frame canonical data.

    Core output rows are retained for the public result API; input coordinates
    and other trajectory fields remain bounded by the shared queue.
    """
    local_request = replace(request, frames=None, every=1)
    pipeline = pipeline or BoundedFramePipeline(resolve_execution_policy(task, request))

    tables = []
    empty = pd.DataFrame()
    with closing(pipeline.map_ordered(selected_frame_envelopes(frames, request), lambda data: task.run(data, local_request).table)) as completed:
        for count, item in enumerate(completed, 1):
            table = item.value.copy()
            empty = table.iloc[:0]
            for column in ("frame_index", "frame_idx"):
                if column in table:
                    table[column] = item.envelope.source_frame
            if not table.empty:
                tables.append(table)
            if reporter:
                reporter("stream", count, 0, "Processing selected frames")
    table = pd.concat(tables, ignore_index=True).infer_objects() if tables else empty
    if not table.empty and sort_columns:
        table = table.sort_values(list(sort_columns), kind="stable").reset_index(drop=True)
    return table
