"""Ordered periodic tracking feeding reusable strain frame kernels."""

from contextlib import closing
import numpy as np

from reaxkit.analysis.stress_strain import common
from reaxkit.core.runtime.execution_contracts import FrameEnvelope


def selected_coordinates(source, eligible, request, pipeline, *, include_reference=False):
    selected = None if request.selected_frames is None else sorted(int(i) for i in request.selected_frames)[::request.every]
    wanted = None if selected is None else set(selected)
    axes = common.periodic_axes(request.periodic)
    identity = None

    def advance(state, item):
        nonlocal identity
        frame, data = item
        if identity is None:
            identity = tuple(data.atom_ids)
        elif identity != tuple(data.atom_ids):
            raise ValueError("Atom identity/order changed during strain streaming.")
        wrapped = np.asarray(data.positions[0, eligible], dtype=float)
        finite = np.isfinite(wrapped).all(axis=1)
        if request.unwrap:
            cells = common.trajectory_cell_matrices(data)
            if cells is None:
                raise ValueError("Periodic unwrapping requires simulation cell lengths.")
            cell = cells[0]
            fractional = np.full_like(wrapped, np.nan)
            fractional[finite] = wrapped[finite] @ np.linalg.inv(cell)
            if state is None:
                if frame != 0:
                    raise ValueError("Strain tracking requires source frame zero.")
                state = (fractional.copy(), fractional.copy(), finite.copy(), frame)
                coordinates = wrapped.copy()
            else:
                previous, unwrapped, valid, last = state
                if frame != last + 1:
                    raise ValueError("Periodic strain requires every intervening source frame.")
                valid &= finite
                step = fractional[valid] - previous[valid]
                for axis, periodic in enumerate(axes):
                    if periodic:
                        step[:, axis] -= np.round(step[:, axis])
                unwrapped[valid] += step
                previous[finite] = fractional[finite]
                coordinates = np.full_like(wrapped, np.nan)
                coordinates[valid] = unwrapped[valid] @ cell
                state = previous, unwrapped, valid, frame
        else:
            coordinates = wrapped
        iteration = int(common.iteration_values(data)[0])
        return state, (frame, coordinates, np.isfinite(coordinates).all(axis=1), iteration)

    envelopes = (FrameEnvelope(index, frame, (frame, data)) for index, (frame, data) in enumerate(source))
    with closing(pipeline.scan_ordered(envelopes, None, advance)) as completed:
        for item in completed:
            frame = item.envelope.source_frame
            if (include_reference and frame == 0) or (frame in wanted if wanted is not None else frame % request.every == 0):
                yield item.value


def run_strain_stream(task, frames, request, calculate, result_type, pipeline=None):
    from reaxkit.core.runtime.execution_contracts import resolve_execution_policy
    from reaxkit.core.runtime.frame_pipeline import BoundedFramePipeline
    from reaxkit.core.runtime.reference_frames import reference_frames
    pipeline = pipeline or BoundedFramePipeline(resolve_execution_policy(task, request))
    if request.every < 1:
        raise ValueError("every must be at least 1.")
    with reference_frames(frames, 0) as (reference, source):
        table = calculate(reference, request, stream=(source, pipeline))
    return result_type(table=table, request=request)
