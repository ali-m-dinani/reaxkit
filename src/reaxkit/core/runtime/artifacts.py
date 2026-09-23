"""Atomic, incremental artifact writing for analysis workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping
from uuid import uuid4

import pandas as pd

ArtifactTier = Literal["core", "summary", "detail", "debug"]
ArtifactFormat = Literal["csv", "parquet", "json"]
OutputProfile = Literal["minimal", "standard", "full", "legacy"]


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    name: str
    filename: str
    tier: ArtifactTier
    default_enabled: bool
    preferred_format: ArtifactFormat
    incremental: bool = False


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
        self.source_frame_min: int | None = None
        self.source_frame_max: int | None = None
        self._parquet_writer = None
        self._closed = False

    def append(self, values: pd.DataFrame | Iterable[Mapping[str, Any]]) -> None:
        if self._closed:
            raise RuntimeError(f"Artifact sink '{self.destination}' is already closed.")
        frame = values if isinstance(values, pd.DataFrame) else pd.DataFrame(list(values))
        if frame.empty:
            return
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        if not self.columns:
            self.columns = [str(value) for value in frame.columns]
            self.dtypes = {str(name): str(dtype) for name, dtype in frame.dtypes.items()}
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

    def finalize(self) -> Path:
        if self._closed:
            return self.destination
        self._closed = True
        if self._parquet_writer is not None:
            self._parquet_writer.close()
        if self.row_count == 0:
            empty = pd.DataFrame(columns=self.columns)
            if self.format_name == "parquet":
                empty.to_parquet(self.temporary, index=False)
            else:
                empty.to_csv(self.temporary, index=False)
        if self.destination.exists() and not self.overwrite:
            self.abort()
            raise FileExistsError(f"Artifact already exists: {self.destination}")
        os.replace(self.temporary, self.destination)
        return self.destination

    def abort(self) -> None:
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None
        self.temporary.unlink(missing_ok=True)
        self._closed = True

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
        overwrite: bool = True,
        manifest_name: str = "reaxkit_artifacts.json",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.profile = profile
        self.overwrite = overwrite
        self.manifest_path = self.output_dir / manifest_name
        self.specs = {spec.name: spec for spec in specs}
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
            self._sinks[name] = BufferedTableSink(
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
        entries: list[dict[str, Any]] = []
        for name, spec in self.specs.items():
            sink = self._sinks.get(name)
            if sink is not None:
                sink.finalize()
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
        payload = {
            "schema_version": 1,
            "output_profile": self.profile,
            "artifacts": entries,
        }
        temporary = self.manifest_path.with_name(f".rk-{uuid4().hex[:8]}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.manifest_path)
        self._finalized = True
        return self.manifest_path

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


__all__ = [
    "ArtifactFormat",
    "ArtifactSpec",
    "ArtifactTier",
    "ArtifactWriter",
    "BufferedTableSink",
    "OutputProfile",
]
