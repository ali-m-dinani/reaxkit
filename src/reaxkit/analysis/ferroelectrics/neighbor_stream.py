"""Shared bounded collection for three- and four-neighbor analyses."""

from contextlib import closing
from dataclasses import replace

import numpy as np
import pandas as pd

from reaxkit.core.runtime.execution_contracts import resolve_execution_policy
from reaxkit.core.runtime.frame_pipeline import BoundedFramePipeline
from reaxkit.core.runtime.frame_tables import selected_frame_envelopes


def stream_neighbors(task, frames, request, result_type, center_columns, neighbor_columns, *, pipeline=None, reporter=None):
    pipeline = pipeline or BoundedFramePipeline(resolve_execution_policy(task, request))
    local = replace(request, frames=None, every=1)
    centers, neighbors, source_frames, iterations = [], [], [], []
    with closing(pipeline.map_ordered(selected_frame_envelopes(frames, request), lambda data: task.run(data, local))) as results:
        for count, completed in enumerate(results, 1):
            result = completed.value
            source = completed.envelope.source_frame
            for table in (result.centers, result.neighbors):
                if not table.empty:
                    table["frame_index"] = source
            centers.append(result.centers)
            neighbors.append(result.neighbors)
            source_frames.append(source)
            iterations.extend(result.iterations.tolist())
            if reporter:
                reporter("stream", count, 0, "Finding neighbors")
    if request.frames is not None:
        missing = set(list(request.frames)[::max(1, int(request.every))]) - set(source_frames)
        if missing:
            raise ValueError(f"Requested frame(s) not found in trajectory: {sorted(missing)}.")
    center_table = pd.concat(centers, ignore_index=True) if centers else pd.DataFrame(columns=center_columns)
    neighbor_table = pd.concat(neighbors, ignore_index=True) if neighbors else pd.DataFrame(columns=neighbor_columns)
    return result_type(table=neighbor_table, request=request, centers=center_table, neighbors=neighbor_table,
                       frame_indices=np.array(source_frames, dtype=int), iterations=np.array(iterations, dtype=int))
