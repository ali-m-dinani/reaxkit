"""Combine primary result tables without retaining their input trajectory."""

from dataclasses import fields, replace
import numpy as np
import pandas as pd
from reaxkit.domain.base_result import BaseResult


def source_result(result, source_frame):
    """Translate a single-frame numerical result to its original source ID."""
    for field in fields(result):
        value = getattr(result, field.name)
        if isinstance(value, pd.DataFrame):
            for column in ("frame_index", "frame_idx", "frame"):
                if column in value:
                    value[column] = source_frame
        elif isinstance(value, BaseResult):
            source_result(value, source_frame)
        elif field.name == "frame_indices":
            setattr(result, field.name, np.full(len(value), source_frame, dtype=int))
    return result


def combine_results(results, request):
    if not results:
        raise ValueError("No selected frames were analyzed.")
    first = results[0]
    values = {}
    for field in fields(first):
        name, value = field.name, getattr(first, field.name)
        if name == "request":
            values[name] = request
        elif isinstance(value, pd.DataFrame) and name != "mapping":
            tables = [getattr(result, name) for result in results]
            values[name] = pd.concat(tables, ignore_index=True).infer_objects()
        elif isinstance(value, BaseResult):
            values[name] = combine_results([getattr(result, name) for result in results], request)
        elif name in {"frame_indices", "iterations"}:
            values[name] = np.concatenate([getattr(result, name) for result in results])
    return replace(first, **values)


class ResultAccumulator:
    """Collect public rows while retaining invariant mappings/reference once.

    Only compact primary tables grow with frame count. Callers spool optional
    atom-level tables before adding a result. Coalescing small DataFrames also
    avoids retaining thousands of pandas managers and their empty views.
    """
    def __init__(self, chunk_frames=128):
        self.first = None
        self.tables = {}
        self.arrays = {}
        self.children = {}
        self.chunk_frames = chunk_frames

    def add(self, result):
        if self.first is None:
            self.first = result
        for field in fields(result):
            name, value = field.name, getattr(result, field.name)
            if isinstance(value, pd.DataFrame) and name != "mapping":
                if name not in self.tables:
                    self.tables[name] = [value.iloc[:0].copy(), [], []]
                prototype, pending, chunks = self.tables[name]
                if not value.empty:
                    pending.append(value)
                    if len(pending) >= self.chunk_frames:
                        chunks.append(pd.concat(pending, ignore_index=True))
                        pending.clear()
            elif isinstance(value, BaseResult):
                child = self.children.setdefault(name, ResultAccumulator(self.chunk_frames))
                child.add(value)
            elif name in {"frame_indices", "iterations"}:
                self.arrays.setdefault(name, []).append(value)

    def finish(self, request):
        if self.first is None:
            raise ValueError("No selected frames were analyzed.")
        values = {"request": request}
        for name, (prototype, pending, chunks) in self.tables.items():
            values[name] = (pd.concat([*chunks, *pending], ignore_index=True).infer_objects()
                            if chunks or pending else prototype)
        values.update({name: np.concatenate(parts) for name, parts in self.arrays.items()})
        values.update({name: child.finish(request) for name, child in self.children.items()})
        return replace(self.first, **values)
