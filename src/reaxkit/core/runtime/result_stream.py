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
