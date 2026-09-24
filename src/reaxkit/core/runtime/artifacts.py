"""Atomic, incremental artifact writing for analysis workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import os
import pickle
from tempfile import TemporaryFile
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping
from uuid import uuid4

import pandas as pd

ArtifactTier = Literal["core", "summary", "detail", "debug"]
ArtifactFormat = Literal["csv", "parquet", "json", "extxyz", "file", "directory"]
OutputProfile = Literal["minimal", "standard", "full", "legacy"]


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    name: str
    filename: str
    tier: ArtifactTier
    default_enabled: bool
    preferred_format: ArtifactFormat
    incremental: bool = False
    units: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TableChunks:
    """A one-pass table producer, consumed only when its artifact is enabled."""

    chunks: Iterable[pd.DataFrame]

    def __iter__(self):
        return iter(self.chunks)

    @property
    def columns(self):
        schema = getattr(self.chunks, "schema", None)
        return () if schema is None else schema.columns


class TableSpool:
    """Replay bounded detail batches from private temporary storage."""

    def __init__(self):
        self.file = TemporaryFile()
        self.schema = pd.DataFrame()

    def append(self, table):
        if self.schema.empty and not len(self.schema.columns):
            self.schema = table.iloc[:0].copy()
        pickle.dump(table, self.file, protocol=pickle.HIGHEST_PROTOCOL)

    def __iter__(self):
        self.file.seek(0)
        while True:
            try:
                yield pickle.load(self.file)
            except EOFError:
                return

    def close(self):
        self.file.close()

    def __del__(self):
        self.close()


class BufferedTableSink:
    """Append DataFrames to one temporary table and publish it atomically."""

    def __init__(self, destination: Path, format_name: ArtifactFormat, *, overwrite: bool) -> None:
        self.destination = destination
        self.format_name = format_name
        self.overwrite = overwrite
        # Keep temporary names short. Analysis output paths can already be near
        # the Windows legacy path limit, and repeating a long destination name
        # plus a full UUID made otherwise valid writes fail.
        self.temporary = destination.with_name(f".rk-{uuid4().hex[:8]}.tmp")
        self.row_count = 0
        self.columns: list[str] = []
        self.dtypes: dict[str, str] = {}
        self._empty_schema = None
        self.source_frame_min: int | None = None
        self.source_frame_max: int | None = None
        self._parquet_writer = None
        self._closed = False

    def append(self, values: pd.DataFrame | Iterable[Mapping[str, Any]]) -> None:
        if self._closed:
            raise RuntimeError(f"Artifact sink '{self.destination}' is already closed.")
        frame = values if isinstance(values, pd.DataFrame) else pd.DataFrame(list(values))
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        if not self.columns:
            self.columns = [str(value) for value in frame.columns]
            self.dtypes = {str(name): str(dtype) for name, dtype in frame.dtypes.items()}
            self._empty_schema = frame.iloc[:0].copy()
        elif list(frame.columns) != self.columns:
            raise ValueError(
                f"Artifact '{self.destination.name}' schema changed between batches: "
                f"expected {self.columns}, received {list(frame.columns)}."
            )
        for column in ("frame_index", "frame_idx", "source_frame"):
            if column in frame.columns:
                numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
                if not numeric.empty:
                    lower, upper = int(numeric.min()), int(numeric.max())
                    self.source_frame_min = lower if self.source_frame_min is None else min(self.source_frame_min, lower)
                    self.source_frame_max = upper if self.source_frame_max is None else max(self.source_frame_max, upper)
                break
        if frame.empty:
            return
        if self.format_name == "csv":
            frame.to_csv(
                self.temporary,
                mode="a",
                header=self.row_count == 0,
                index=False,
            )
        elif self.format_name == "parquet":
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError(
                    "Incremental Parquet output requires pyarrow. Install the parquet extra "
                    "or choose CSV for the optional detail artifact."
                ) from exc
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if self._parquet_writer is None:
                self._parquet_writer = pq.ParquetWriter(self.temporary, table.schema)
            self._parquet_writer.write_table(table)
        else:
            raise ValueError(f"Buffered table output does not support format '{self.format_name}'.")
        self.row_count += len(frame)

    def prepare(self) -> None:
        """Close the temporary table before any artifact is published."""
        if self._closed:
            return
        self._closed = True
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None
        if self.row_count == 0:
            empty = self._empty_schema if self._empty_schema is not None else pd.DataFrame(columns=self.columns)
            if self.format_name == "parquet":
                empty.to_parquet(self.temporary, index=False)
            else:
                empty.to_csv(self.temporary, index=False)
        if self.destination.exists() and not self.overwrite:
            self.abort()
            raise FileExistsError(f"Artifact already exists: {self.destination}")

    def finalize(self) -> Path:
        self.prepare()
        if not self.temporary.exists():
            return self.destination
        if self.overwrite:
            os.replace(self.temporary, self.destination)
        else:
            try:
                os.link(self.temporary, self.destination)
            finally:
                self.temporary.unlink(missing_ok=True)
        return self.destination

    def abort(self) -> None:
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None
        self.temporary.unlink(missing_ok=True)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.finalize()
        else:
            self.abort()
        return False

    def manifest_entry(self, spec: ArtifactSpec) -> dict[str, Any]:
        return {
            **asdict(spec),
            "path": str(self.destination),
            "row_count": self.row_count,
            "columns": self.columns,
            "dtypes": self.dtypes,
            "source_frame_min": self.source_frame_min,
            "source_frame_max": self.source_frame_max,
            "status": "written",
        }


class ArtifactWriter:
    """Apply output policy and atomically publish declared artifacts."""

    def __init__(
        self,
        output_dir: str | Path,
        specs: Iterable[ArtifactSpec],
        *,
        profile: OutputProfile = "standard",
        overwrite: bool = False,
        manifest_name: str = "reaxkit_artifacts.json",
        metadata: Mapping[str, Any] | None = None,
        detail_format: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.profile = profile
        self.overwrite = overwrite
        self.manifest_path = self.output_dir / manifest_name
        if profile not in {"standard", "minimal", "full", "legacy"}:
            raise ValueError(f"Unknown output profile: {profile}")
        if detail_format not in {None, "csv", "parquet"}:
            raise ValueError(f"Unknown detail format: {detail_format}")
        self.metadata = dict(metadata or {})
        self.specs = {}
        filenames = {manifest_name}
        for spec in specs:
            if spec.tier in {"detail", "debug"} and spec.preferred_format in {"csv", "parquet"} and (detail_format or profile == "legacy"):
                format_name = detail_format or "csv"
                spec = replace(spec, preferred_format=format_name,
                               filename=str(Path(spec.filename).with_suffix(f".{format_name}")))
            path = Path(spec.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Artifact filename must stay inside output directory: {path}")
            if spec.name in self.specs or spec.filename in filenames:
                raise ValueError(f"Duplicate artifact name or filename: {spec.name}")
            self.specs[spec.name] = spec
            filenames.add(spec.filename)
        self._sinks: dict[str, BufferedTableSink] = {}
        self._registered: dict[str, dict[str, Any]] = {}
        self._finalized = False

    def enabled(self, name: str) -> bool:
        spec = self.specs[name]
        if self.profile in {"full", "legacy"}:
            return True
        if self.profile == "minimal":
            return spec.tier == "core"
        return spec.default_enabled

    def sink(self, name: str) -> BufferedTableSink | None:
        if not self.enabled(name):
            return None
        if name not in self._sinks:
            spec = self.specs[name]
            sink_class = AtomicFileSink if spec.preferred_format in {"json", "extxyz", "file"} else BufferedTableSink
            self._sinks[name] = sink_class(
                self.output_dir / spec.filename,
                spec.preferred_format,
                overwrite=self.overwrite,
            )
        return self._sinks[name]

    def append(self, name: str, values: pd.DataFrame | Iterable[Mapping[str, Any]]) -> None:
        sink = self.sink(name)
        if sink is not None:
            sink.append(values)

    def write_table(self, name: str, table: pd.DataFrame) -> None:
        self.append(name, table)

    def write_json(self, name: str, payload: Any) -> None:
        if self.specs[name].preferred_format != "json":
            raise ValueError(f"Artifact '{name}' is not declared as JSON.")
        self.write_file(name, lambda path: path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"))

    def write_file(self, name: str, producer) -> None:
        """Let a file producer write privately, publishing only on success."""
        sink = self.sink(name)
        if sink is None:
            return
        if not isinstance(sink, AtomicFileSink):
            raise ValueError("Use write_table or write_chunks for tabular artifacts.")
        sink.destination.parent.mkdir(parents=True, exist_ok=True)
        producer(sink.temporary)

    def write_chunks(self, name: str, chunks: TableChunks) -> None:
        if not self.enabled(name):
            return
        iterator = iter(chunks)
        try:
            for table in iterator:
                self.append(name, table)
        finally:
            close = getattr(iterator, "close", None)
            if close:
                close()

    def register_file(self, name: str, path: str | Path, **metadata: Any) -> None:
        self._registered[name] = {
            "name": name,
            "path": str(Path(path)),
            "status": "written",
            **metadata,
        }

    def finalize(self) -> Path:
        if self._finalized:
            return self.manifest_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_name(f".rk-{uuid4().hex[:8]}.tmp")
        backups: dict[Path, Path] = {}
        published: list[Path] = []
        try:
            if self.manifest_path.exists() and self.metadata.get("command"):
                existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                owner = existing.get("run_metadata", {}).get("command")
                if owner and owner != self.metadata["command"]:
                    raise FileExistsError(f"Output directory belongs to command '{owner}': {self.output_dir}")
            if self.manifest_path.exists() and not self.overwrite:
                raise FileExistsError(f"Artifact manifest already exists: {self.manifest_path}")
            for sink in self._sinks.values():
                sink.prepare()
            payload = self._manifest_payload()
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            publications = [(sink.temporary, sink.destination) for sink in self._sinks.values()]
            publications.append((temporary, self.manifest_path))
            for source, destination in publications:
                if not self.overwrite:
                    # Hard-link creation fails atomically if another run won
                    # the destination, unlike an exists()/replace() sequence.
                    os.link(source, destination)
                    published.append(destination)
                    source.unlink()
                    continue
                if destination.exists():
                    backup = destination.with_name(f".rk-{uuid4().hex[:8]}.bak")
                    os.replace(destination, backup)
                    backups[destination] = backup
                os.replace(source, destination)
                published.append(destination)
        except BaseException:
            for destination in reversed(published):
                destination.unlink(missing_ok=True)
            for destination, backup in backups.items():
                os.replace(backup, destination)
            temporary.unlink(missing_ok=True)
            self.abort()
            raise
        for backup in backups.values():
            backup.unlink(missing_ok=True)
        self._finalized = True
        return self.manifest_path

    def _manifest_payload(self) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        for name, spec in self.specs.items():
            sink = self._sinks.get(name)
            if sink is not None:
                entries.append(sink.manifest_entry(spec))
            elif name in self._registered:
                entries.append({**asdict(spec), **self._registered[name]})
            else:
                entries.append({
                    **asdict(spec),
                    "path": str(self.output_dir / spec.filename),
                    "status": "omitted",
                    "reason": "disabled_by_output_profile" if not self.enabled(name) else "no_rows",
                })
        return {
            "schema_version": 1,
            "output_profile": self.profile,
            "artifacts": entries,
            "run_metadata": self.metadata,
        }

    def abort(self) -> None:
        for sink in self._sinks.values():
            sink.abort()

    def __enter__(self) -> "ArtifactWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is None:
            self.finalize()
        else:
            self.abort()
        return False


class AtomicFileSink(BufferedTableSink):
    """Publication adapter for JSON and incrementally generated extxyz files."""

    def append(self, values):
        raise TypeError("Use write_file or write_json for this artifact.")

    def prepare(self):
        if not self.temporary.is_file():
            raise FileNotFoundError(f"Artifact producer did not create '{self.destination.name}'.")
        if self.destination.exists() and not self.overwrite:
            raise FileExistsError(f"Artifact already exists: {self.destination}")
        self._closed = True

    def manifest_entry(self, spec):
        return {**asdict(spec), "path": str(self.destination), "status": "written",
                "bytes": self.temporary.stat().st_size}


__all__ = [
    "ArtifactFormat",
    "ArtifactSpec",
    "ArtifactTier",
    "ArtifactWriter",
    "BufferedTableSink",
    "OutputProfile",
    "TableChunks",
]
