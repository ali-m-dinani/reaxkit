"""Ordered bond-event recovery including smoothing and hysteresis state."""

import pandas as pd

from reaxkit.core.runtime.frame_tables import selected_frame_envelopes
from reaxkit.core.runtime.result_store import SortedResultTable


def stream_bond_events(frames, request, reporter=None):
    from reaxkit.analysis.connectivity.connectivity import ConnectionListTask, ConnectionListRequest, BondEventsResult
    from reaxkit.analysis.connectivity.stream_state import BondTraceState, EVENT_COLUMNS
    store = request._result_store
    state = store.scientific_state() or {"traces": {}, "finalized": False}
    traces = {}
    for key, saved in state["traces"].items():
        trace = BondTraceState(request)
        trace.restore(saved)
        traces[key] = trace
    local = ConnectionListRequest(frames=None, every=1, min_bo=0.0, undirected=request.undirected, include_self=False)
    target = None
    if request.src is not None and request.dst is not None:
        target = (int(request.src), int(request.dst))
        if request.undirected:
            target = tuple(sorted(target))
    dtypes = {name: "int64" if name in {"source", "destination", "frame_idx", "iter"}
              else "float64" if name in {"bo_at_event", "threshold", "hysteresis"} else "str" for name in EVENT_COLUMNS}

    def commit(pending, finalized=False):
        position, source, rows = pending
        table = pd.DataFrame(rows, columns=EVENT_COLUMNS).astype(dtypes)
        snapshot = {"traces": {key: trace.snapshot() for key, trace in traces.items()}, "finalized": finalized}
        store.append(position, source, {"table": table}, state=snapshot)

    def record(rows, key, events):
        for frame, iteration, value, event in events:
            rows.append((*key, event, frame, iteration, value, float(request.threshold), float(request.hysteresis)))

    pending = None
    if not state["finalized"]:
        committed = store.committed_frames
        for position, envelope in enumerate(selected_frame_envelopes(frames, request)):
            if position < committed:
                continue
            if pending is not None:
                commit(pending)
            rows = []
            table = ConnectionListTask().run(envelope.payload, local).table
            for row in table.itertuples(index=False):
                if target is not None and (row.source, row.destination) != target:
                    continue
                key = (row.source, row.source_type, row.destination, row.destination_type)
                trace = traces.setdefault(key, BondTraceState(request))
                record(rows, key, trace.add(envelope.source_frame, row.iteration, row.BO))
            pending = position, envelope.source_frame, rows
            if reporter:
                reporter("stream", position + 1, 0, "Checkpointing ordered bond events")
        if pending is not None:
            for key, trace in traces.items():
                record(pending[2], key, trace.finish())
            commit(pending, finalized=True)
    store.transition("analysis_complete")
    table = SortedResultTable(store, "table", ("source", "destination", "iter", "event"))
    return BondEventsResult(table=pd.DataFrame(columns=EVENT_COLUMNS) if table.empty else table, request=request)
