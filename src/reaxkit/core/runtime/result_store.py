"""Durable, single-writer result batches independent of the input cache."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import time
from uuid import uuid4

import pandas as pd


SCHEMA_VERSION = 1
STATES = {"running", "failed", "interrupted", "analysis_complete", "complete"}


def _json_write(path, value):
    temporary = path.with_name(f".{uuid4().hex[:8]}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _sync_directory(path):
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(paths):
    """Hash resolved inputs, conservatively rejecting replacement or append."""
    identities = []
    for path in sorted({Path(path).resolve() for path in paths}):
        before = path.stat()
        checksum = _digest(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Input changed while computing identity: {path}")
        identities.append({"path": str(path), "size": after.st_size, "sha256": checksum})
    return identities


class ResultStore:
    """Commit all tables for ordered frames with an atomic manifest boundary.

    The manifest contains a batch count, not an ever-growing list of frames.
    Published directories beyond that count are uncommitted orphans. Buffers
    are bounded by bytes, frames and elapsed time, plus one oversized frame.
    """

    def __init__(self, directory, identity, *, resume=False, buffer_bytes=32 * 1024 * 1024,
                 batch_frames=128, interval_seconds=30, format_name="parquet", fault=None):
        if buffer_bytes <= 0 or batch_frames <= 0 or not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("Checkpoint buffer, frame threshold and interval must be positive.")
        if format_name not in {"parquet", "csv"}:
            raise ValueError("Checkpoint table format must be parquet or csv.")
        if format_name == "parquet":
            try:
                import pyarrow
            except ImportError:
                format_name = "csv"
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "manifest.json"
        self.lock_path = self.directory / "writer.lock"
        self.token = uuid4().hex
        self._locked = False
        self.pending = []
        self.pending_bytes = 0
        self.pending_state = None
        self.buffer_bytes = int(buffer_bytes)
        self.batch_frames = int(batch_frames)
        self.interval_seconds = float(interval_seconds)
        self.last_flush = time.monotonic()
        self.fault = fault or (lambda stage: None)
        try:
            with self.lock_path.open("x", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "host": socket.gethostname(), "token": self.token}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            self._locked = True
        except FileExistsError as exc:
            raise RuntimeError(f"Checkpoint has a writer lock: {self.lock_path}. Verify the writer has stopped before explicit stale-lock removal.") from exc
        try:
            normalized = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
            if resume:
                self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if self.manifest["schema_version"] != SCHEMA_VERSION or self.manifest["identity"] != normalized:
                    raise ValueError("Checkpoint inputs, scientific settings or versions are incompatible.")
                self.validate()
                if not self.committed_frames:
                    self.manifest["tables"] = {}
                    self.manifest.pop("result_descriptor", None)
            else:
                if self.manifest_path.exists():
                    raise FileExistsError(f"Checkpoint already exists; resume explicitly: {self.directory}")
                self.manifest = {"schema_version": SCHEMA_VERSION, "identity": normalized,
                                 "state": "running", "batch_count": 0, "frame_count": 0,
                                 "last_position": None, "last_source_frame": None,
                                 "tables": {}, "format": format_name, "null_token": f"__null_{uuid4().hex}__"}
                _json_write(self.manifest_path, self.manifest)
            (self.directory / "batches").mkdir(exist_ok=True)
        except BaseException:
            self.close()
            raise

    def _owned(self):
        if not self._locked:
            raise RuntimeError("Checkpoint writer is closed.")

    @property
    def committed_frames(self):
        return self.manifest["frame_count"]

    def _batch_path(self, index):
        return self.directory / "batches" / f"batch-{index:08d}"

    def batches(self):
        for index in range(self.manifest["batch_count"]):
            path = self._batch_path(index)
            yield path, json.loads((path / "batch.json").read_text(encoding="utf-8"))

    def validate(self):
        frames = 0
        previous = -1
        previous_digest = None
        counts = {name: 0 for name in self.manifest["tables"]}
        for path, metadata in self.batches():
            if metadata.get("previous_digest") != previous_digest:
                raise ValueError(f"Damaged checkpoint batch metadata: {path}")
            previous_digest = _digest(path / "batch.json")
            if "state_sha256" in metadata and _digest(path / "state.json") != metadata["state_sha256"]:
                raise ValueError(f"Damaged checkpoint scientific state: {path}")
            if set(metadata["tables"]) != set(counts):
                raise ValueError(f"Incomplete checkpoint batch: {path}")
            for position, source in metadata["frames"]:
                if position <= previous:
                    raise ValueError("Checkpoint selection positions are not strictly increasing.")
                previous = position
                frames += 1
            for name, entry in metadata["tables"].items():
                data_path = path / entry["file"]
                if data_path.parent != path or _digest(data_path) != entry["sha256"]:
                    raise ValueError(f"Damaged checkpoint table: {data_path}")
                table = self._read(data_path)
                if len(table) != entry["rows"] or list(table.columns) != self.manifest["tables"][name]["columns"]:
                    raise ValueError(f"Invalid checkpoint schema/counts: {data_path}")
                counts[name] += len(table)
        if frames != self.committed_frames or (frames and previous != self.manifest["last_position"]):
            raise ValueError("Checkpoint frame counts do not match the manifest.")
        if previous_digest != self.manifest.get("last_batch_digest"):
            raise ValueError("Damaged checkpoint batch metadata chain.")
        if any(counts[name] != schema["rows"] for name, schema in self.manifest["tables"].items()):
            raise ValueError("Checkpoint row counts do not match the manifest.")

    def _read(self, path):
        schema = next(value for value in self.manifest["tables"].values() if value["file"] == path.name)
        if path.suffix == ".parquet":
            table = pd.read_parquet(path)
            return table.astype(dict(zip(schema["columns"], schema["dtypes"]))) if table.empty else table
        if not schema["columns"]:
            return pd.DataFrame()
        return pd.read_csv(path, dtype=dict(zip(schema["columns"], schema["dtypes"])), float_precision="round_trip",
                           keep_default_na=False, na_values=[self.manifest["null_token"]])

    def append(self, position, source_frame, tables, *, state=None):
        self._owned()
        tables = dict(tables)
        if self.manifest["state"] != "running":
            raise RuntimeError("Only running checkpoints accept result frames.")
        previous = self.pending[-1][0] if self.pending else self.manifest["last_position"]
        if previous is not None and int(position) <= previous:
            raise ValueError("Result selection positions must increase without duplicates.")
        schemas = self.manifest["tables"]
        if schemas and set(tables) != set(schemas):
            raise ValueError("Every enabled table must be present in each frame transaction.")
        for name, table in tables.items():
            if not isinstance(table, pd.DataFrame) or not table.columns.is_unique:
                raise ValueError("Checkpoint tables require DataFrames with unique columns.")
            if any(not isinstance(column, str) for column in table.columns):
                raise ValueError("Checkpoint column names must be strings.")
            schema = {"columns": list(table.columns), "dtypes": [str(dtype) for dtype in table.dtypes]}
            if name in schemas:
                if table.empty and schema["columns"] == schemas[name]["columns"]:
                    table = tables[name] = table.astype(dict(zip(schema["columns"], schemas[name]["dtypes"])))
                    schema["dtypes"] = schemas[name]["dtypes"]
                elif schemas[name]["rows"] == 0 and all(parts[name].empty for _, _, parts in self.pending):
                    schemas[name]["dtypes"] = schema["dtypes"]
                if schema["columns"] != schemas[name]["columns"] or schema["dtypes"] != schemas[name]["dtypes"]:
                    raise ValueError(f"Checkpoint schema changed for {name}.")
            else:
                schemas[name] = {**schema, "rows": 0, "file": f"table-{len(schemas):03d}.{self.manifest['format']}"}
        size = sum(int(table.memory_usage(index=True, deep=True).sum()) for table in tables.values())
        if self.pending and self.pending_bytes + size > self.buffer_bytes:
            self.flush()
        self.pending.append((int(position), int(source_frame), {name: table.copy(deep=True) for name, table in tables.items()}))
        if state is not None:
            from reaxkit.core.runtime.checkpoint_results import encode
            self.pending_state = encode(state)
        self.pending_bytes += size
        if (self.pending_bytes >= self.buffer_bytes or len(self.pending) >= self.batch_frames
                or time.monotonic() - self.last_flush >= self.interval_seconds):
            self.flush()

    def flush(self):
        self._owned()
        if not self.pending:
            return
        staging = self.directory / f".s-{uuid4().hex[:8]}"
        staging.mkdir()
        updated = json.loads(json.dumps(self.manifest))
        metadata = {"frames": [[position, source] for position, source, _ in self.pending], "tables": {},
                    "previous_digest": self.manifest.get("last_batch_digest")}
        try:
            for name, schema in updated["tables"].items():
                table = pd.concat([tables[name] for _, _, tables in self.pending], ignore_index=True)
                path = staging / schema["file"]
                if self.manifest["format"] == "parquet":
                    table.to_parquet(path, index=False)
                else:
                    token = self.manifest["null_token"]
                    if table.astype(str).eq(token).any().any():
                        raise ValueError("Checkpoint CSV null token conflicts with table data.")
                    table.to_csv(path, index=False, na_rep=token)
                with path.open("rb+") as stream:
                    os.fsync(stream.fileno())
                self.fault("writing")
                metadata["tables"][name] = {"file": path.name, "rows": len(table), "sha256": _digest(path)}
                schema["rows"] += len(table)
            if self.pending_state is not None:
                _json_write(staging / "state.json", self.pending_state)
                metadata["state_sha256"] = _digest(staging / "state.json")
            _json_write(staging / "batch.json", metadata)
            updated["last_batch_digest"] = _digest(staging / "batch.json")
            self.fault("staged")
            destination = self._batch_path(updated["batch_count"])
            if destination.exists():
                orphan = self.directory / f"orphan-{uuid4().hex[:8]}"
                os.replace(destination, orphan)
            os.replace(staging, destination)
            _sync_directory(destination.parent)
            self.fault("published")
            updated["batch_count"] += 1
            updated["frame_count"] += len(self.pending)
            updated["last_position"], updated["last_source_frame"] = metadata["frames"][-1]
            self.fault("manifest")
            _json_write(self.manifest_path, updated)
            self.manifest = updated
            self.pending.clear()
            self.pending_bytes = 0
            self.pending_state = None
            self.last_flush = time.monotonic()
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def transition(self, state, *, error=None, failed_source_frame=None):
        self._owned()
        if state not in STATES:
            raise ValueError(f"Unknown checkpoint state: {state}")
        if state in {"analysis_complete", "complete"}:
            self.flush()
            if not self.committed_frames:
                raise ValueError("No selected frames were analyzed.")
        if state == "complete" and self.manifest["state"] != "analysis_complete":
            raise ValueError("Final publication requires completed analysis.")
        updated = {**self.manifest, "state": state, "error": error, "failed_source_frame": failed_source_frame}
        _json_write(self.manifest_path, updated)
        self.manifest = updated

    def table(self, name):
        return ResultTable(self, name)

    def scientific_state(self):
        if not self.manifest["batch_count"]:
            return None
        from reaxkit.core.runtime.checkpoint_results import decode
        path = self._batch_path(self.manifest["batch_count"] - 1) / "state.json"
        return decode(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else None

    def close(self):
        if self._locked:
            owner = json.loads(self.lock_path.read_text(encoding="utf-8"))
            if owner["token"] != self.token:
                raise RuntimeError("Checkpoint lock ownership changed.")
            self.lock_path.unlink()
            self._locked = False

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        try:
            if error is not None:
                self.transition("interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed", error=str(error))
        finally:
            self.close()


class ResultTable:
    """Explicit disk-backed table access; never implicitly materialize history."""

    def __init__(self, store, name, transform=None):
        self.store = store
        self.name = name
        self.transform = transform

    @property
    def columns(self):
        schema = self.store.manifest["tables"][self.name]
        columns = pd.Index(schema["columns"])
        if self.transform is not None:
            prototype = pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in zip(columns, schema["dtypes"])})
            return self.transform(prototype).columns
        return columns

    def __len__(self):
        return self.store.manifest["tables"][self.name]["rows"]

    @property
    def empty(self):
        return len(self) == 0

    def __iter__(self):
        for path, metadata in self.store.batches():
            table = self.store._read(path / metadata["tables"][self.name]["file"])
            yield self.transform(table) if self.transform is not None else table

    def map_batches(self, transform):
        previous = self.transform
        return ResultTable(self.store, self.name, lambda table: transform(previous(table) if previous else table))

    def materialize(self):
        parts = list(self)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=self.columns)

    def select_frame(self, source_frame, column="frame_index"):
        parts = []
        for path, metadata in self.store.batches():
            if source_frame not in (source for _, source in metadata["frames"]):
                continue
            table = self.store._read(path / metadata["tables"][self.name]["file"])
            if self.transform is not None:
                table = self.transform(table)
            selected = table.loc[table[column] == source_frame]
            if not selected.empty:
                parts.append(selected)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=self.columns)

    def to_csv(self, path, *, index=False):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{uuid4().hex[:8]}.tmp")
        try:
            with temporary.open("w", newline="", encoding="utf-8") as stream:
                for number, table in enumerate(self):
                    table.to_csv(stream, index=index, header=number == 0)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def table_chunks(table):
    return iter(table) if isinstance(table, ResultTable) else iter((table,))


def frame_table(table, source, column="frame_index"):
    return table.select_frame(source, column) if isinstance(table, ResultTable) else table.loc[table[column] == source]


def finite_extrema(table, columns):
    import numpy as np
    lower = np.full(len(columns), np.inf)
    upper = np.full(len(columns), -np.inf)
    for chunk in table_chunks(table):
        for index, column in enumerate(columns):
            values = chunk[column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values):
                lower[index] = min(lower[index], values.min())
                upper[index] = max(upper[index], values.max())
    return lower, upper


class SortedResultTable(ResultTable):
    """Stable external sorting for results whose ordering spans batches."""

    def __init__(self, store, name, sort_columns):
        super().__init__(store, name)
        self.sort_columns = tuple(sort_columns)

    def map_batches(self, transform):
        result = SortedResultTable(self.store, self.name, self.sort_columns)
        previous = self.transform
        result.transform = lambda table: transform(previous(table) if previous else table)
        return result

    def __iter__(self):
        import sqlite3
        from tempfile import TemporaryDirectory
        schema = self.store.manifest["tables"][self.name]
        with TemporaryDirectory(prefix="reaxkit-sort-") as directory:
            connection = sqlite3.connect(str(Path(directory) / "rows.sqlite"))
            try:
                connection.execute("PRAGMA cache_size=-8192")
                connection.execute("PRAGMA temp_store=FILE")
                for table in super().__iter__():
                    table.to_sql("results", connection, if_exists="append", index=False)
                quoted = [f'"{name.replace(chr(34), chr(34) * 2)}"' for name in self.sort_columns]
                query = 'SELECT * FROM results ORDER BY ' + ', '.join([*quoted, 'rowid'])
                for table in pd.read_sql_query(query, connection, chunksize=8192):
                    yield table.astype(dict(zip(schema["columns"], schema["dtypes"])))
            finally:
                connection.close()

    def select_frame(self, source_frame, column="frame_index"):
        parts = []
        for table in self:
            selected = table.loc[table[column] == source_frame]
            if not selected.empty:
                parts.append(selected)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=self.columns)
