"""Versioned result descriptors and bounded dataclass result accumulation."""

from dataclasses import fields, is_dataclass
import importlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

from reaxkit.core.runtime.result_store import _json_write, _digest


def encode(value):
    if isinstance(value, np.ndarray):
        return {"kind": "array", "dtype": str(value.dtype), "values": encode(value.tolist())}
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, Path):
        return {"kind": "path", "value": str(value)}
    if isinstance(value, pd.DataFrame):
        return {"kind": "dataframe", "columns": list(value.columns),
                "dtypes": [str(dtype) for dtype in value.dtypes], "values": encode(value.to_numpy().tolist())}
    if is_dataclass(value):
        return {"kind": "dataclass", "module": type(value).__module__, "class": type(value).__name__,
                "fields": {field.name: encode(getattr(value, field.name)) for field in fields(value)}}
    if type(value).__module__.startswith("ase.") and type(value).__name__ == "Atoms":
        return {"kind": "atoms", "numbers": value.numbers.tolist(), "positions": value.positions.tolist(),
                "cell": value.cell.array.tolist(), "pbc": value.pbc.tolist()}
    if isinstance(value, (tuple, list)):
        return {"kind": "tuple" if isinstance(value, tuple) else "list", "values": [encode(item) for item in value]}
    if isinstance(value, dict):
        return {"kind": "dict", "values": [[encode(key), encode(item)] for key, item in value.items()]}
    if isinstance(value, float) and not math.isfinite(value):
        return {"kind": "nonfinite", "value": str(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported checkpoint state: {type(value).__name__}")


def decode(value):
    if not isinstance(value, dict):
        return value
    kind = value["kind"]
    if kind == "nonfinite":
        return float(value["value"])
    if kind == "array":
        return np.asarray(decode(value["values"]), dtype=value["dtype"])
    if kind == "path":
        return Path(value["value"])
    if kind == "dataframe":
        return pd.DataFrame(decode(value["values"]), columns=value["columns"]).astype(dict(zip(value["columns"], value["dtypes"])))
    if kind == "atoms":
        from ase import Atoms
        return Atoms(numbers=value["numbers"], positions=value["positions"], cell=value["cell"], pbc=value["pbc"])
    if kind == "dataclass":
        if not value["module"].startswith(("reaxkit.analysis.", "reaxkit.domain.")):
            raise ValueError("Unsupported checkpoint result class.")
        cls = getattr(importlib.import_module(value["module"]), value["class"])
        if not is_dataclass(cls):
            raise ValueError("Checkpoint descriptor must name a dataclass.")
        return cls(**{name: decode(item) for name, item in value["fields"].items()})
    if kind in {"tuple", "list"}:
        values = [decode(item) for item in value["values"]]
        return tuple(values) if kind == "tuple" else values
    if kind == "dict":
        return {decode(key): decode(item) for key, item in value["values"]}
    raise ValueError(f"Unsupported checkpoint descriptor: {kind}")


class CheckpointAccumulator:
    """Store result tables and rebuild the result with explicit table references."""

    def __init__(self, store):
        self.store = store
        self.descriptor = None
        if store.manifest.get("result_descriptor"):
            import json
            path = store.directory / "result.json"
            if _digest(path) != store.manifest["result_descriptor"]:
                raise ValueError("Damaged checkpoint result descriptor.")
            self.descriptor = json.loads(path.read_text(encoding="utf-8"))

    def add(self, result, position, source_frame, *, extra_tables=None):
        tables = dict(extra_tables or {})

        def visit(value, path):
            if path.split(".")[-1] == "time_values":
                values = np.full(len(result.frame_indices), np.nan) if value is None else np.asarray(value, dtype=float)
                tables[path] = pd.DataFrame({"value": values})
                return {"kind": "optional_array", "name": path}
            if isinstance(value, pd.DataFrame) and path.split(".")[-1] != "mapping":
                tables[path] = value
                return {"kind": "table", "name": path}
            if isinstance(value, np.ndarray) and path.split(".")[-1] in {"frame_indices", "iterations", "time_values"}:
                tables[path] = pd.DataFrame({"value": value})
                return {"kind": "result_array", "name": path, "dtype": str(value.dtype)}
            if is_dataclass(value) and path.split(".")[-1] != "request":
                from reaxkit.domain.base_result import BaseResult
                if isinstance(value, BaseResult):
                    return {"kind": "result", "module": type(value).__module__, "class": type(value).__name__,
                            "fields": {field.name: visit(getattr(value, field.name), f"{path}.{field.name}".strip(".")) for field in fields(value)}}
            if isinstance(value, dict) and value and all(isinstance(item, pd.DataFrame) for item in value.values()):
                return {"kind": "table_dict", "values": {key: visit(item, f"{path}.{key}") for key, item in value.items()}}
            return encode(value)

        descriptor = visit(result, "")
        if self.descriptor is None:
            _json_write(self.store.directory / "result.json", descriptor)
            self.store.manifest["result_descriptor"] = _digest(self.store.directory / "result.json")
            self.descriptor = descriptor
        self.store.append(position, source_frame, tables)

    def finish(self, request):
        self.store.transition("analysis_complete")
        if self.descriptor is None:
            raise ValueError("No selected frames were analyzed.")

        def build(value):
            if not isinstance(value, dict):
                return value
            kind = value["kind"]
            if kind == "table":
                return self.store.table(value["name"])
            if kind == "result_array":
                return self.store.table(value["name"]).materialize()["value"].to_numpy(dtype=value["dtype"])
            if kind == "optional_array":
                values = self.store.table(value["name"]).materialize()["value"].to_numpy(dtype=float)
                return values if np.isfinite(values).all() else None
            if kind == "table_dict":
                return {key: build(item) for key, item in value["values"].items()}
            if kind == "result":
                if not value["module"].startswith("reaxkit.analysis."):
                    raise ValueError("Unsupported checkpoint result class.")
                cls = getattr(importlib.import_module(value["module"]), value["class"])
                if not is_dataclass(cls):
                    raise ValueError("Checkpoint result must be a dataclass.")
                return cls(**{name: request if name == "request" else build(item) for name, item in value["fields"].items()})
            return decode(value)

        result = build(self.descriptor)
        result.skip_result_cache = True
        result.checkpoint_directory = str(self.store.directory)
        return result


def trajectory_table(trajectory, source):
    positions = np.asarray(trajectory.positions[0])
    table = pd.DataFrame(positions, columns=["x", "y", "z"])
    table["frame_index"] = source
    table["atom_id"] = trajectory.atom_ids
    table["element"] = trajectory.elements
    table["label"] = trajectory.elements if trajectory.atom_labels is None else trajectory.atom_labels[0]
    table["iteration"] = source if trajectory.iterations is None else int(trajectory.iterations[0])
    simulation = trajectory.simulation
    for field in ("cell_lengths", "cell_angles"):
        values = getattr(simulation, field, None)
        for axis in range(3):
            table[f"{field}_{axis}"] = np.nan if values is None else float(values[0][axis])
    times = getattr(simulation, "time", None)
    table["time"] = np.nan if times is None else float(times[0])
    return table


def restore_trajectory(result, store):
    if "_trajectory" not in store.manifest["tables"]:
        return result
    from reaxkit.core.runtime.trajectory_spool import TrajectorySpool
    from reaxkit.domain.data_models import SimulationData, TrajectoryData
    spool = TrajectorySpool()
    try:
        for table in store.table("_trajectory"):
            for source, group in table.groupby("frame_index", sort=False):
                first = group.iloc[0]
                lengths = np.array([[first[f"cell_lengths_{axis}"] for axis in range(3)]])
                angles = np.array([[first[f"cell_angles_{axis}"] for axis in range(3)]])
                iterations = np.array([first["iteration"]], dtype=int)
                ids = group.atom_id.tolist()
                simulation = SimulationData(atom_ids=ids, iterations=iterations,
                    cell_lengths=lengths if np.isfinite(lengths).all() else None,
                    cell_angles=angles if np.isfinite(angles).all() else None,
                    time=np.array([first["time"]]) if np.isfinite(first["time"]) else None)
                trajectory = TrajectoryData(positions=group[["x", "y", "z"]].to_numpy()[None],
                    elements=group.element.tolist(), atom_ids=ids, iterations=iterations,
                    atom_labels=group.label.to_numpy()[None], simulation=simulation)
                spool.append(int(source), trajectory)
        result.trajectory = spool.finish()
        result._trajectory_spool = spool
    except BaseException:
        spool.close()
        raise
    return result


def materialize_for_global_presentation(result):
    """Explicit global plot/report boundary, separate from bounded export."""
    from dataclasses import replace
    from reaxkit.domain.base_result import BaseResult
    from reaxkit.core.runtime.result_store import ResultTable
    def convert(value):
        if isinstance(value, ResultTable):
            return value.materialize()
        if isinstance(value, BaseResult):
            return replace(value, **{field.name: convert(getattr(value, field.name)) for field in fields(value)})
        if isinstance(value, dict):
            return {name: convert(item) for name, item in value.items()}
        return value
    return convert(result)
