"""Ordered lifetime segmentation with state only for observed species."""

from contextlib import closing
import pandas as pd
from reaxkit.core.runtime.frame_tables import selected_frame_envelopes
from reaxkit.core.runtime.execution_contracts import resolve_execution_policy
from reaxkit.core.runtime.frame_pipeline import BoundedFramePipeline

COLUMNS = ["molecular_formula", "lifetime_segment_id", "start_frame_index", "end_frame_index",
           "start_iter", "end_iter", "lifetime_segment_sampled_step_count", "peak_freq", "mean_freq"]


def lifetime_table(task, frames, request, pipeline=None, reporter=None):
    pipeline = pipeline or BoundedFramePipeline(resolve_execution_policy(task, request))
    active, identifiers, rows = {}, {}, []
    known = set(str(value) for value in request.molecules) if request.molecules is not None else set()
    first = previous = None
    count = 0

    def finish(formula):
        record = active.pop(formula)
        rows.append({"molecular_formula": formula, "lifetime_segment_id": identifiers[formula],
                     "start_frame_index": record[0], "end_frame_index": record[2],
                     "start_iter": record[1], "end_iter": record[3],
                     "lifetime_segment_sampled_step_count": record[4], "peak_freq": record[5],
                     "mean_freq": record[6] / record[4]})

    def step(state, data):
        nonlocal first, previous, count
        frame, payload = data
        iteration = int(payload.iterations[0])
        first = first or (frame, iteration)
        species = payload.molecular_species
        if species.molecular_formula.duplicated().any():
            raise ValueError("Duplicate molecular formulas in a single iteration.")
        values = dict(zip(species.molecular_formula.astype(str), pd.to_numeric(species.freq, errors="coerce").fillna(0.0)))
        if request.molecules is None:
            for formula in set(values) - known:
                if count and request.min_freq <= 0:
                    identifiers[formula] = 1
                    active[formula] = [*first, *previous, count, 0.0, 0.0, 0.0]
            known.update(values)
        for formula in sorted(known):
            value = float(values.get(formula, 0.0))
            if value < request.min_freq:
                if formula in active:
                    finish(formula)
                continue
            if formula not in active:
                identifiers[formula] = identifiers.get(formula, 0) + 1
                active[formula] = [frame, iteration, frame, iteration, 0, value, 0.0, 0.0]
            record = active[formula]
            record[2:4] = frame, iteration
            record[4] += 1
            record[5] = max(record[5], value)
            corrected = value - record[7]
            updated = record[6] + corrected
            record[7] = (updated - record[6]) - corrected
            record[6] = updated
        previous = frame, iteration
        count += 1
        if reporter:
            reporter("stream", count, 0, "Tracking molecule lifetimes")
        return state, None

    def source():
        for envelope in selected_frame_envelopes(frames, request):
            envelope.payload = (envelope.source_frame, envelope.payload)
            yield envelope
    with closing(pipeline.scan_ordered(source(), None, step)) as results:
        for _ in results:
            pass
    for formula in sorted(active):
        finish(formula)
    return pd.DataFrame(rows, columns=COLUMNS).sort_values(["molecular_formula", "start_iter"], kind="stable").reset_index(drop=True)
