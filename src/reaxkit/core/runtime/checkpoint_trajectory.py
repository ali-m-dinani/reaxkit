"""Commit complete Extended XYZ frames together with their scientific tables."""

from dataclasses import replace
from io import StringIO
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from reaxkit.core.runtime.checkpoint_results import CheckpointAccumulator
from reaxkit.engine.common.generators.extended_xyz_generator import ExtendedXYZWriter


def segment_table(frame, precision):
    writer = ExtendedXYZWriter("unused.extxyz", precision=precision)
    writer._handle = StringIO()
    writer.write_frame(frame)
    text = writer._handle.getvalue()
    writer._handle.close()
    if len(text.splitlines()) != len(frame.species) + 2:
        raise ValueError("Extended XYZ frame contains invalid embedded line breaks.")
    return pd.DataFrame({"frame_index": [int(frame.frame)], "text": [text], "atom_rows": [len(frame.species)]})


def publish_segments(store, output):
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".xyz-{uuid4().hex[:8]}.tmp")
    rows = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for table in store.table("_extxyz"):
                for text in table["text"]:
                    stream.write(text)
                rows += int(table["atom_rows"].sum())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return rows


def stream_charge_trajectory(frames, request, reporter=None):
    from reaxkit.analysis.ferroelectrics.charge_extxyz import (
        _validate_request, _source_frame, _baseline_by_atom, _electric_field_metadata,
        _frame_iteration, _result, charge_extended_xyz_frame,
    )
    _validate_request(request)
    store = request._result_store
    durable = CheckpointAccumulator(store)
    if store.manifest["state"] not in {"analysis_complete", "complete"}:
        baseline = None
        requested = None if request.frames is None else set(request.frames[::int(request.every)])
        seen = set()
        position = 0
        for index, data in enumerate(frames):
            source = _source_frame(data, index)
            if requested is not None:
                seen.add(source)
            if source == 0:
                baseline = _baseline_by_atom(data)
            if source not in requested if requested is not None else source % int(request.every) != 0:
                continue
            position += 1
            if position <= store.committed_frames:
                continue
            if baseline is None:
                raise ValueError("Frame 0 is required to calculate delta_charge.")
            extended = charge_extended_xyz_frame(data, baseline, source_frame=source,
                metadata=_electric_field_metadata(data, request, _frame_iteration(data)))
            result = _result(request, [source], [int(extended.iteration)], [len(extended.species)])
            durable.add(result, position - 1, source, extra_tables={"_extxyz": segment_table(extended, request.precision)})
            if reporter:
                reporter("stream", position, 0, "Checkpointing charge trajectory frames")
        if requested is not None and requested - seen:
            raise ValueError(f"Requested frames not found: {sorted(requested - seen)}")
    result = durable.finish(request)
    rows = publish_segments(store, request._output_path)
    result.output_path = str(Path(request._output_path).resolve())
    result.table = pd.DataFrame([{"output_path": result.output_path,
                                 "frames_written": store.committed_frames, "atom_rows_written": rows}])
    return result


def stream_potential_trajectory(frames, request, reporter=None):
    from reaxkit.analysis.electrostatics.potential_and_electric_field.analysis import (
        _source_frame, calculate_potential_and_field, material_midpoint,
    )
    from reaxkit.analysis.electrostatics.potential_and_electric_field.trajectory import (
        _extended_frame, PotentialElectricFieldTrajectoryResult,
    )
    if not request._output_path or int(request.precision) < 1:
        raise ValueError("A trajectory output path and precision >= 1 are required.")
    store = request._result_store
    durable = CheckpointAccumulator(store)
    if store.manifest["state"] not in {"analysis_complete", "complete"}:
        requested = None if request.frames is None else set(request.frames[::int(request.every)])
        seen = set()
        position = 0
        fixed = request.potential_reference_position
        for data in frames:
            source = _source_frame(data.trajectory, 0)
            if requested is not None:
                seen.add(source)
            if source not in requested if requested is not None else source % int(request.every) != 0:
                continue
            if fixed is None and request.potential_reference_mode == "fixed-midpoint":
                positions = np.asarray(data.trajectory.positions[0])
                fixed = tuple(material_midpoint(positions[np.isfinite(positions).all(axis=1)]))
            position += 1
            if position <= store.committed_frames:
                continue
            local = replace(request, frames=[0], every=1, potential_reference_position=fixed)
            result = calculate_potential_and_field(data, local, preserve_source_indices=True)
            extended = _extended_frame(data, local, result)
            durable.add(result, position - 1, source, extra_tables={"_extxyz": segment_table(extended, request.precision)})
            if reporter:
                reporter("stream", position, 0, "Checkpointing potential/field trajectory frames")
        if requested is not None and requested - seen:
            raise ValueError(f"Requested frames not found: {sorted(requested - seen)}")
    calculated = durable.finish(request)
    rows = publish_segments(store, request._output_path)
    output = str(Path(request._output_path).resolve())
    result = PotentialElectricFieldTrajectoryResult(
        pd.DataFrame([{"output_path": output, "frames_written": store.committed_frames, "atom_rows_written": rows}]),
        request, output, calculated.frame_indices, calculated.iterations, calculated)
    result.skip_result_cache = True
    result.checkpoint_directory = str(store.directory)
    return result
