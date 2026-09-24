"""Persistent source-frame cache shared by analysis commands.

The frame store is intentionally independent of engines and analysis tasks. A
file handler supplies a source identity, a parse-view identity, optional byte
offsets, and parsed frame payloads. Later commands can then reuse compatible
records without using the command name or analysis request as part of the key.

SQLite is part of the Python standard library, provides atomic transactions,
and avoids creating one filesystem entry per cached frame. ReaxFF xmolout and
fort.7 handlers use this store for explicit finite selections.
"""

from __future__ import annotations

from contextlib import closing, contextmanager
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import sqlite3
import threading
from time import perf_counter, monotonic
from typing import Any, Iterable, Mapping, Sequence

FRAME_STORE_SCHEMA_VERSION = 1
FRAME_PAYLOAD_VERSION = "1"
SOURCE_IDENTITY_VERSION = "1"
DEFAULT_SIGNATURE_BLOCK_SIZE = 64 * 1024
DEFAULT_BUSY_TIMEOUT_SECONDS = 5.0
DEFAULT_FRAME_CACHE_MAX_BYTES = 10 * 1024 ** 3
_SQLITE_QUERY_BATCH_SIZE = 500
_INDEX_LOCK = threading.Lock()
_MAINTENANCE_BYTES = 64 * 1024 ** 2
_MAINTENANCE_SECONDS = 30.0


@contextmanager
def _generation_guard(root: Path, generation: Path, *, exclusive=False, timeout=5.0):
    """Cross-process lease, released by SQLite even if the owning process dies.

    The small rollback-journal database lives outside the evictable generation.
    Shared read transactions pin it; eviction holds an exclusive transaction
    through deletion. Payload transactions use a different database.
    """
    key = sha256(str(generation.resolve()).encode("utf-8")).hexdigest()
    path = root / "index" / "leases" / f"{key}.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=timeout, isolation_level=None)) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS lease (id INTEGER PRIMARY KEY)")
        connection.execute("BEGIN EXCLUSIVE" if exclusive else "BEGIN")
        connection.execute("SELECT id FROM lease").fetchall()
        try:
            yield
        finally:
            connection.rollback()


def _leased(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.session():
            return method(self, *args, **kwargs)
    return wrapped


class FrameStoreError(RuntimeError):
    """Base error for an unusable frame store."""


class FrameStoreVersionError(FrameStoreError):
    """Raised when an existing store uses an unsupported schema version."""


class FrameStoreIdentityError(FrameStoreError):
    """Raised when store metadata does not match its requested identity."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _normalize_identity_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _normalize_identity_value(value: Any) -> Any:
    """Return a deterministic JSON-compatible cache-identity value."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Cache identity floats must be finite.")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_identity_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_identity_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize_identity_value(item) for item in value]
        return sorted(normalized, key=_canonical_json)
    raise TypeError(
        "Cache identity values must use JSON-compatible primitives, paths, "
        f"mappings, or sequences; received {type(value).__name__}."
    )


def _hash_json(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _safe_namespace(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_.-]+", "-", str(value).strip().lower())
    return normalized.strip("-._") or "unknown"


def _bounded_file_signature(path: Path, *, block_size: int) -> str:
    """Hash bounded samples from a source without scanning the complete file."""
    if block_size <= 0:
        raise ValueError("Signature block size must be positive.")
    size = path.stat().st_size
    digest = sha256()
    digest.update(str(int(size)).encode("ascii"))
    digest.update(b"\0")
    with path.open("rb") as handle:
        first = handle.read(block_size)
        digest.update(first)
        if size > block_size:
            tail_start = max(block_size, size - block_size)
            handle.seek(tail_start)
            digest.update(b"\0")
            digest.update(str(int(tail_start)).encode("ascii"))
            digest.update(b"\0")
            digest.update(handle.read(block_size))
    return digest.hexdigest()


@dataclass(frozen=True)
class FrameSourceIdentity:
    """Stable source path identity plus a validated file generation."""

    engine: str
    source_kind: str
    path: str
    size: int
    mtime_ns: int
    signature: str
    identity_version: str = SOURCE_IDENTITY_VERSION

    @classmethod
    def from_path(
            cls,
            path: str | Path,
            *,
            engine: str,
            source_kind: str,
            signature_block_size: int = DEFAULT_SIGNATURE_BLOCK_SIZE,
    ) -> "FrameSourceIdentity":
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Frame-cache source is not a file: {resolved}")

        before = resolved.stat()
        signature = _bounded_file_signature(
            resolved,
            block_size=int(signature_block_size),
        )
        after = resolved.stat()
        if (
                int(before.st_size) != int(after.st_size)
                or int(before.st_mtime_ns) != int(after.st_mtime_ns)
        ):
            raise FrameStoreError(
                f"Frame-cache source changed while its identity was being read: {resolved}"
            )

        return cls(
            engine=_safe_namespace(engine),
            source_kind=_safe_namespace(source_kind),
            path=os.path.normcase(str(resolved)),
            size=int(after.st_size),
            mtime_ns=int(after.st_mtime_ns),
            signature=signature,
        )

    @property
    def source_key(self) -> str:
        """Identify the logical source independently of its current contents."""
        return _hash_json(
            {
                "engine": self.engine,
                "source_kind": self.source_kind,
                "path": self.path,
                "identity_version": self.identity_version,
            }
        )

    @property
    def generation_key(self) -> str:
        """Identify one validated generation of the logical source."""
        return _hash_json(
            {
                "source_key": self.source_key,
                "size": self.size,
                "mtime_ns": self.mtime_ns,
                "signature": self.signature,
                "identity_version": self.identity_version,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "source_kind": self.source_kind,
            "path": self.path,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "signature": self.signature,
            "identity_version": self.identity_version,
            "source_key": self.source_key,
            "generation_key": self.generation_key,
        }


@dataclass(frozen=True)
class FrameViewIdentity:
    """Parser/output contract for one compatible family of frame payloads.

    ``representation`` names the payload schema. ``capabilities`` describe the
    information present in that schema. A consumer may use a cached view only
    when :meth:`can_satisfy` returns true; narrower payloads are never promoted
    implicitly to richer requests.
    """

    parser: str
    parser_version: str
    representation: str
    capabilities: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)
    _options_json: str = field(init=False, compare=True, repr=False)

    def __post_init__(self) -> None:
        parser = str(self.parser).strip()
        parser_version = str(self.parser_version).strip()
        representation = str(self.representation).strip()
        if not parser or not parser_version or not representation:
            raise ValueError("Parser, parser version, and representation cannot be empty.")
        capabilities = tuple(sorted({str(item).strip() for item in self.capabilities if str(item).strip()}))
        options_json = _canonical_json(dict(self.options))
        object.__setattr__(self, "parser", parser)
        object.__setattr__(self, "parser_version", parser_version)
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "_options_json", options_json)

    @property
    def normalized_options(self) -> dict[str, Any]:
        return json.loads(self._options_json)

    @property
    def view_key(self) -> str:
        return _hash_json(self.as_dict())

    def can_satisfy(
            self,
            *,
            representation: str | None = None,
            capabilities: Iterable[str] = (),
    ) -> bool:
        if representation is not None and str(representation) != self.representation:
            return False
        required = {str(item) for item in capabilities}
        return required.issubset(set(self.capabilities))

    def as_dict(self) -> dict[str, Any]:
        return {
            "parser": self.parser,
            "parser_version": self.parser_version,
            "representation": self.representation,
            "capabilities": list(self.capabilities),
            "options": self.normalized_options,
        }


@dataclass(frozen=True)
class InputCachePolicy:
    """Input-cache policy, intentionally separate from analysis-result caching."""

    enabled: bool = True

    @classmethod
    def from_args(cls, args: Mapping[str, Any] | None) -> "InputCachePolicy":
        values = args or {}
        enabled = bool(values.get("input_cache", True))
        if bool(values.get("no_input_cache", False)):
            enabled = False
        return cls(enabled=enabled)


@dataclass(frozen=True)
class FrameOffset:
    """Byte range and lightweight header data for one source frame."""

    frame_index: int
    byte_start: int
    byte_end: int
    iteration: int | None = None
    atom_count: int | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("Frame index cannot be negative.")
        if self.byte_start < 0 or self.byte_end < self.byte_start:
            raise ValueError("Frame byte range must satisfy 0 <= start <= end.")
        if self.atom_count is not None and self.atom_count < 0:
            raise ValueError("Frame atom count cannot be negative.")


@dataclass(frozen=True)
class IndexCoverage:
    """Safe resume point for an incremental frame-offset scan."""

    next_frame_index: int = 0
    next_byte_offset: int = 0
    complete: bool = False

    def __post_init__(self) -> None:
        if self.next_frame_index < 0 or self.next_byte_offset < 0:
            raise ValueError("Index coverage values cannot be negative.")


class FrameStore:
    """Versioned SQLite store for offsets and parsed source-frame payloads."""

    def __init__(
            self,
            cache_root: str | Path,
            *,
            source: FrameSourceIdentity,
            view: FrameViewIdentity,
            busy_timeout_seconds: float = DEFAULT_BUSY_TIMEOUT_SECONDS,
            max_bytes: int | None = None,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.source = source
        self.view = view
        self.busy_timeout_seconds = max(0.0, float(busy_timeout_seconds))
        self.max_bytes = max_bytes
        self._session_connection = None
        self._session_depth = 0
        self._growth = 0
        self._known_bytes = 0
        self._last_maintenance = float("-inf")
        self._last_touch = float("-inf")
        self._write_suspended = False
        self.stats = {"cache_read_seconds": 0.0, "cache_write_seconds": 0.0,
                      "cache_metadata_seconds": 0.0, "cache_offsets_written": 0,
                      "cache_maintenance_seconds": 0.0, "cache_maintenance_calls": 0,
                      "cache_payload_bytes_written": 0, "cache_payload_bytes_read": 0,
                      "cache_commits": 0, "cache_frames_written": 0,
                      "cache_writes_skipped": 0}
        namespace = f"{source.engine}-{source.source_kind}"
        self.path = (
                self.cache_root
                / "frames"
                / namespace
                / source.source_key[:20]
                / source.generation_key[:20]
                / f"{view.view_key[:20]}.sqlite3"
        )
        with _generation_guard(self.cache_root, self.path.parent):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()
            self._update_workspace_index()

    @classmethod
    def for_source(
            cls,
            cache_root: str | Path,
            source_path: str | Path,
            *,
            engine: str,
            source_kind: str,
            parser: str,
            parser_version: str,
            representation: str,
            capabilities: Iterable[str] = (),
            options: Mapping[str, Any] | None = None,
            signature_block_size: int = DEFAULT_SIGNATURE_BLOCK_SIZE,
            busy_timeout_seconds: float = DEFAULT_BUSY_TIMEOUT_SECONDS,
            max_bytes: int | None = None,
    ) -> "FrameStore":
        source = FrameSourceIdentity.from_path(
            source_path,
            engine=engine,
            source_kind=source_kind,
            signature_block_size=signature_block_size,
        )
        view = FrameViewIdentity(
            parser=parser,
            parser_version=parser_version,
            representation=representation,
            capabilities=tuple(capabilities),
            options=dict(options or {}),
        )
        if max_bytes is None:
            configured = os.environ.get("REAXKIT_FRAME_CACHE_MAX_BYTES", "").strip()
            max_bytes = int(configured) if configured else DEFAULT_FRAME_CACHE_MAX_BYTES
        return cls(
            cache_root,
            source=source,
            view=view,
            busy_timeout_seconds=busy_timeout_seconds,
            max_bytes=max_bytes,
        )

    @property
    def index_path(self) -> Path:
        return self.cache_root / "index" / "frames.json"

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_seconds * 1000)}")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    @contextmanager
    def _connection(self):
        if self._session_connection is not None:
            yield self._session_connection
        else:
            with closing(self._connect()) as connection:
                yield connection

    @contextmanager
    def session(self):
        """Pin a generation and amortize connections over one reader stream.

        A session belongs to its calling thread. Use separate stores for
        concurrent readers/writers. Memory and transactions remain bounded.
        """
        if self._session_depth:
            yield self
            return
        with _generation_guard(self.cache_root, self.path.parent,
                               timeout=self.busy_timeout_seconds):
            if not self.path.exists():
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._initialize()
                self._update_workspace_index()
            self._session_connection = self._connect()
            self._session_depth = 1
            try:
                yield self
            finally:
                self._session_depth = 0
                self._session_connection.close()
                self._session_connection = None
        # Quota maintenance runs after releasing our lease; other streams are
        # still protected by theirs. Do not mask an analysis error on cleanup.
        if self._growth and self.max_bytes and self.max_bytes > 0:
            self._maintain_quota(force=True)

    def _maintain_quota(self, *, force=False, incoming=0):
        if not self.max_bytes or self.max_bytes <= 0:
            return True
        now = monotonic()
        interval = min(_MAINTENANCE_BYTES, max(1, self.max_bytes // 16))
        due = (force or now - self._last_maintenance >= _MAINTENANCE_SECONDS
               or self._growth >= interval)
        if self._write_suspended and not due:
            return False
        if due or self._known_bytes + self._growth + incoming > self.max_bytes:
            started = perf_counter()
            # Leave space for this batch; pinned generations are never removed.
            target = max(1, int(self.max_bytes) - incoming)
            try:
                info = enforce_frame_cache_limit(self.cache_root, target)
            except (OSError, sqlite3.DatabaseError):
                self._write_suspended = True
                self._last_maintenance = now
                return False
            finally:
                self.stats["cache_maintenance_seconds"] += perf_counter() - started
                self.stats["cache_maintenance_calls"] += 1
            self._known_bytes = int(info["bytes_after"])
            self._growth = 0
            self._last_maintenance = now
        self._write_suspended = self._known_bytes + self._growth + incoming > self.max_bytes
        return not self._write_suspended

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in {0, FRAME_STORE_SCHEMA_VERSION}:
                connection.rollback()
                raise FrameStoreVersionError(
                    f"Unsupported frame-store schema version {version}; "
                    f"expected {FRAME_STORE_SCHEMA_VERSION}: {self.path}"
                )
            statements = (
                """
                CREATE TABLE IF NOT EXISTS metadata
                (
                    key
                    TEXT
                    PRIMARY
                    KEY,
                    value
                    TEXT
                    NOT
                    NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS frame_offsets
                (
                    frame_index
                    INTEGER
                    PRIMARY
                    KEY
                    CHECK
                (
                    frame_index
                    >=
                    0
                ),
                    byte_start INTEGER NOT NULL CHECK
                (
                    byte_start
                    >=
                    0
                ),
                    byte_end INTEGER NOT NULL CHECK
                (
                    byte_end
                    >=
                    byte_start
                ),
                    iteration INTEGER,
                    atom_count INTEGER CHECK
                (
                    atom_count
                    IS
                    NULL
                    OR
                    atom_count
                    >=
                    0
                ),
                    updated_at TEXT NOT NULL
                    )
                """,
                """
                CREATE TABLE IF NOT EXISTS frames
                (
                    frame_index
                    INTEGER
                    NOT
                    NULL
                    CHECK
                (
                    frame_index
                    >=
                    0
                ),
                    representation TEXT NOT NULL,
                    payload_version TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    PRIMARY KEY
                (
                    frame_index,
                    representation
                )
                    )
                """,
                """
                CREATE TABLE IF NOT EXISTS index_coverage
                (
                    singleton
                    INTEGER
                    PRIMARY
                    KEY
                    CHECK
                (
                    singleton =
                    1
                ),
                    next_frame_index INTEGER NOT NULL CHECK
                (
                    next_frame_index
                    >=
                    0
                ),
                    next_byte_offset INTEGER NOT NULL CHECK
                (
                    next_byte_offset
                    >=
                    0
                ),
                    complete INTEGER NOT NULL CHECK
                (
                    complete
                    IN
                (
                    0,
                    1
                )),
                    updated_at TEXT NOT NULL
                    )
                """,
            )
            for statement in statements:
                connection.execute(statement)
            # Additive migration: COUNT(*) uses the small primary-key index;
            # never scan the BLOB rows to recover last_accessed_at.
            connection.execute("CREATE TABLE IF NOT EXISTS cache_summary ("
                               "id INTEGER PRIMARY KEY CHECK(id=1), frame_count INTEGER NOT NULL, "
                               "last_accessed_at TEXT NOT NULL)")
            if connection.execute("SELECT 1 FROM cache_summary WHERE id=1").fetchone() is None:
                count = connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
                connection.execute("INSERT INTO cache_summary VALUES (1, ?, ?)",
                                   (count, _utc_now_iso()))
            connection.execute("CREATE TRIGGER IF NOT EXISTS frame_count_insert AFTER INSERT ON frames "
                               "BEGIN UPDATE cache_summary SET frame_count=frame_count+1 WHERE id=1; END")
            connection.execute("CREATE TRIGGER IF NOT EXISTS frame_count_delete AFTER DELETE ON frames "
                               "BEGIN UPDATE cache_summary SET frame_count=frame_count-1 WHERE id=1; END")
            connection.execute(f"PRAGMA user_version = {FRAME_STORE_SCHEMA_VERSION}")
            expected = {
                "schema_version": str(FRAME_STORE_SCHEMA_VERSION),
                "payload_version": FRAME_PAYLOAD_VERSION,
                "source": _canonical_json(self.source.as_dict()),
                "view": _canonical_json(self.view.as_dict()),
            }
            existing = {
                str(row["key"]): str(row["value"])
                for row in connection.execute("SELECT key, value FROM metadata")
            }
            for key, value in expected.items():
                current = existing.get(key)
                if current is not None and current != value:
                    connection.rollback()
                    raise FrameStoreIdentityError(
                        f"Frame-store metadata mismatch for {key!r}: {self.path}"
                    )
                connection.execute(
                    "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)",
                    (key, value),
                )
            connection.execute(
                """
                INSERT
                OR IGNORE INTO index_coverage(
                    singleton, next_frame_index, next_byte_offset, complete, updated_at
                ) VALUES (1, 0, 0, 0, ?)
                """,
                (_utc_now_iso(),),
            )
            connection.commit()

    def _update_workspace_index(self) -> None:
        entry_key = sha256(str(self.path.resolve()).encode("utf-8")).hexdigest()
        entry = {
            "path": str(self.path),
            "engine": self.source.engine,
            "source_kind": self.source.source_kind,
            "source_path": self.source.path,
            "source_key": self.source.source_key,
            "generation_key": self.source.generation_key,
            "view_key": self.view.view_key,
            "representation": self.view.representation,
            "updated_at": _utc_now_iso(),
        }
        with _INDEX_LOCK:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            payload: dict[str, Any] = {}
            if self.index_path.exists():
                try:
                    payload = json.loads(self.index_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    payload = {}
            payload.setdefault("namespace", "frames")
            payload.setdefault("schema_version", 1)
            payload.setdefault("entries", {})
            payload["entries"][entry_key] = entry
            tmp = self.index_path.with_name(
                f"{self.index_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            try:
                tmp.write_text(
                    json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                tmp.replace(self.index_path)
            finally:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _normalize_indices(indices: Iterable[int]) -> tuple[int, ...]:
        normalized: list[int] = []
        seen: set[int] = set()
        for raw in indices:
            value = int(raw)
            if value < 0:
                raise ValueError("Frame indices cannot be negative.")
            if value not in seen:
                seen.add(value)
                normalized.append(value)
        return tuple(normalized)

    @staticmethod
    def _batches(values: Sequence[int]) -> Iterable[Sequence[int]]:
        for start in range(0, len(values), _SQLITE_QUERY_BATCH_SIZE):
            yield values[start: start + _SQLITE_QUERY_BATCH_SIZE]

    @_leased
    def put_frames(self, frames: Mapping[int, Any]) -> set[int]:
        """Persist serializable frames and return the indices successfully stored."""
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()
            self._update_workspace_index()
        now = _utc_now_iso()
        prepared: list[tuple[int, str, str, bytes, str, str, str]] = []
        serialized_at = perf_counter()
        for raw_index, value in frames.items():
            frame_index = int(raw_index)
            if frame_index < 0:
                raise ValueError("Frame indices cannot be negative.")
            try:
                payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                continue
            prepared.append(
                (
                    frame_index,
                    self.view.representation,
                    FRAME_PAYLOAD_VERSION,
                    payload,
                    sha256(payload).hexdigest(),
                    now,
                    now,
                )
            )
        self.stats["cache_write_seconds"] += perf_counter() - serialized_at
        if not prepared:
            return set()
        payload_bytes = sum(len(row[3]) for row in prepared)
        # Conservative allowance for SQLite pages and metadata. An oversized
        # optional batch is skipped, never allowed to evict its active reader.
        reservation = payload_bytes + len(prepared) * 8192
        if not self._maintain_quota(incoming=reservation):
            self.stats["cache_writes_skipped"] += len(prepared)
            return set()
        started = perf_counter()
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.executemany(
                    """
                    INSERT INTO frames(frame_index, representation, payload_version, payload,
                                       checksum, created_at, last_accessed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(frame_index, representation) DO
                    UPDATE SET
                        payload_version = excluded.payload_version,
                        payload = excluded.payload,
                        checksum = excluded.checksum,
                        last_accessed_at = excluded.last_accessed_at
                    """,
                    prepared,
                )
                connection.execute("UPDATE cache_summary SET last_accessed_at=? WHERE id=1", (now,))
                connection.commit()
        except (OSError, sqlite3.DatabaseError):
            if self._session_connection is not None:
                self._session_connection.rollback()
            return set()
        finally:
            self.stats["cache_write_seconds"] += perf_counter() - started
        self._growth += reservation
        self.stats["cache_payload_bytes_written"] += payload_bytes
        self.stats["cache_frames_written"] += len(prepared)
        self.stats["cache_commits"] += 1
        return {row[0] for row in prepared}

    @_leased
    def get_frames(self, indices: Iterable[int]) -> dict[int, Any]:
        """Return valid cached frames; missing or corrupt entries are omitted."""
        requested = self._normalize_indices(indices)
        if not requested:
            return {}
        rows: dict[int, sqlite3.Row] = {}
        started = perf_counter()
        try:
            with self._connection() as connection:
                for batch in self._batches(requested):
                    placeholders = ",".join("?" for _ in batch)
                    query = (
                        "SELECT frame_index, payload_version, payload, checksum "
                        "FROM frames WHERE representation = ? "
                        f"AND frame_index IN ({placeholders})"
                    )
                    params: tuple[Any, ...] = (self.view.representation, *batch)
                    for row in connection.execute(query, params):
                        rows[int(row["frame_index"])] = row
        except (OSError, sqlite3.DatabaseError):
            return {}

        loaded: dict[int, Any] = {}
        invalid: list[int] = []
        for frame_index in requested:
            row = rows.get(frame_index)
            if row is None:
                continue
            if str(row["payload_version"]) != FRAME_PAYLOAD_VERSION:
                invalid.append(frame_index)
                continue
            payload = bytes(row["payload"])
            self.stats["cache_payload_bytes_read"] += len(payload)
            if sha256(payload).hexdigest() != str(row["checksum"]):
                invalid.append(frame_index)
                continue
            try:
                loaded[frame_index] = pickle.loads(payload)
            except Exception:
                invalid.append(frame_index)

        self._touch_and_remove_invalid(tuple(loaded), tuple(invalid))
        self.stats["cache_read_seconds"] += perf_counter() - started
        return loaded

    def _touch_and_remove_invalid(
            self,
            valid_indices: Sequence[int],
            invalid_indices: Sequence[int],
    ) -> None:
        if not valid_indices and not invalid_indices:
            return
        touch = monotonic() - self._last_touch >= _MAINTENANCE_SECONDS
        if not invalid_indices and not touch:
            return
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                now = _utc_now_iso()
                if touch:
                    connection.execute("UPDATE cache_summary SET last_accessed_at=? WHERE id=1", (now,))
                for batch in self._batches(invalid_indices):
                    placeholders = ",".join("?" for _ in batch)
                    connection.execute(
                        "DELETE FROM frames WHERE representation = ? "
                        f"AND frame_index IN ({placeholders})",
                        (self.view.representation, *batch),
                    )
                connection.commit()
                self._last_touch = monotonic()
                self.stats["cache_commits"] += 1
        except (OSError, sqlite3.DatabaseError):
            if self._session_connection is not None:
                self._session_connection.rollback()
            return

    @_leased
    def available_indices(self, indices: Iterable[int] | None = None) -> set[int]:
        """Return stored indices without deserializing frame payloads."""
        try:
            with self._connection() as connection:
                if indices is None:
                    rows = connection.execute(
                        "SELECT frame_index FROM frames WHERE representation = ?",
                        (self.view.representation,),
                    )
                    return {int(row["frame_index"]) for row in rows}
                requested = self._normalize_indices(indices)
                available: set[int] = set()
                for batch in self._batches(requested):
                    placeholders = ",".join("?" for _ in batch)
                    rows = connection.execute(
                        "SELECT frame_index FROM frames WHERE representation = ? "
                        f"AND frame_index IN ({placeholders})",
                        (self.view.representation, *batch),
                    )
                    available.update(int(row["frame_index"]) for row in rows)
                return available
        except (OSError, sqlite3.DatabaseError):
            return set()

    def missing_indices(self, indices: Iterable[int]) -> tuple[int, ...]:
        requested = self._normalize_indices(indices)
        available = self.available_indices(requested)
        return tuple(frame_index for frame_index in requested if frame_index not in available)

    @_leased
    def put_offsets(self, offsets: Iterable[FrameOffset]) -> set[int]:
        values = tuple(offsets)
        if not values:
            return set()
        now = _utc_now_iso()
        rows = [
            (
                item.frame_index,
                item.byte_start,
                item.byte_end,
                item.iteration,
                item.atom_count,
                now,
            )
            for item in values
        ]
        reservation = 8192 + len(rows) * 256
        # Offset metadata is needed for indexed cache misses even when optional
        # payload writes are suspended. Account for its growth at reconciliation.
        started = perf_counter()
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.executemany(
                    """
                    INSERT INTO frame_offsets(frame_index, byte_start, byte_end, iteration, atom_count, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(frame_index) DO
                    UPDATE SET
                        byte_start = excluded.byte_start,
                        byte_end = excluded.byte_end,
                        iteration = excluded.iteration,
                        atom_count = excluded.atom_count,
                        updated_at = excluded.updated_at
                    """,
                    rows,
                )
                connection.commit()
        except (OSError, sqlite3.DatabaseError):
            if self._session_connection is not None:
                self._session_connection.rollback()
            return set()
        finally:
            self.stats["cache_metadata_seconds"] += perf_counter() - started
        self._growth += reservation
        self.stats["cache_offsets_written"] += len(values)
        self.stats["cache_commits"] += 1
        return {item.frame_index for item in values}

    @_leased
    def get_offsets(self, indices: Iterable[int]) -> dict[int, FrameOffset]:
        requested = self._normalize_indices(indices)
        if not requested:
            return {}
        result: dict[int, FrameOffset] = {}
        try:
            with self._connection() as connection:
                for batch in self._batches(requested):
                    placeholders = ",".join("?" for _ in batch)
                    for row in connection.execute(
                            "SELECT frame_index, byte_start, byte_end, iteration, atom_count "
                            f"FROM frame_offsets WHERE frame_index IN ({placeholders})",
                            tuple(batch),
                    ):
                        item = FrameOffset(
                            frame_index=int(row["frame_index"]),
                            byte_start=int(row["byte_start"]),
                            byte_end=int(row["byte_end"]),
                            iteration=None if row["iteration"] is None else int(row["iteration"]),
                            atom_count=None if row["atom_count"] is None else int(row["atom_count"]),
                        )
                        result[item.frame_index] = item
        except (OSError, sqlite3.DatabaseError):
            return {}
        return result

    @_leased
    def get_coverage(self) -> IndexCoverage:
        try:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT next_frame_index, next_byte_offset, complete "
                    "FROM index_coverage WHERE singleton = 1"
                ).fetchone()
        except (OSError, sqlite3.DatabaseError):
            return IndexCoverage()
        if row is None:
            return IndexCoverage()
        return IndexCoverage(
            next_frame_index=int(row["next_frame_index"]),
            next_byte_offset=int(row["next_byte_offset"]),
            complete=bool(row["complete"]),
        )

    @_leased
    def set_coverage(self, coverage: IndexCoverage) -> bool:
        started = perf_counter()
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                current = connection.execute(
                    "SELECT next_frame_index, next_byte_offset, complete "
                    "FROM index_coverage WHERE singleton = 1"
                ).fetchone()
                if current is not None:
                    current_frame = int(current["next_frame_index"])
                    current_offset = int(current["next_byte_offset"])
                    current_complete = bool(current["complete"])
                    if (
                            coverage.next_frame_index < current_frame
                            or coverage.next_byte_offset < current_offset
                            or (current_complete and not coverage.complete)
                    ):
                        connection.rollback()
                        return False
                connection.execute(
                    """
                    INSERT INTO index_coverage(singleton, next_frame_index, next_byte_offset, complete, updated_at)
                    VALUES (1, ?, ?, ?, ?) ON CONFLICT(singleton) DO
                    UPDATE SET
                        next_frame_index = excluded.next_frame_index,
                        next_byte_offset = excluded.next_byte_offset,
                        complete = excluded.complete,
                        updated_at = excluded.updated_at
                    """,
                    (
                        coverage.next_frame_index,
                        coverage.next_byte_offset,
                        int(coverage.complete),
                        _utc_now_iso(),
                    ),
                )
                connection.commit()
                self.stats["cache_commits"] += 1
                return True
        except (OSError, sqlite3.DatabaseError):
            if self._session_connection is not None:
                self._session_connection.rollback()
            return False
        finally:
            self.stats["cache_metadata_seconds"] += perf_counter() - started


def inspect_frame_cache(cache_root: str | Path, *, details: bool = True) -> dict[str, Any]:
    """Return deterministic size and entry information for a workspace frame cache."""
    root = Path(cache_root)
    frames_root = root / "frames"
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    if frames_root.is_dir():
        for path in sorted(frames_root.rglob("*.sqlite3"), key=lambda item: str(item)):
            size = 0
            for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
                try:
                    size += int(candidate.stat().st_size)
                except OSError:
                    pass
            total_bytes += size
            try:
                last_accessed = str(path.stat().st_mtime_ns)
            except OSError:
                last_accessed = ""
            frame_count = 0
            if details:
                try:
                    with closing(sqlite3.connect(path, timeout=0.1)) as connection:
                        if connection.execute("SELECT 1 FROM sqlite_master WHERE name='cache_summary'").fetchone():
                            row = connection.execute(
                                "SELECT frame_count, last_accessed_at FROM cache_summary WHERE id=1"
                            ).fetchone()
                            frame_count, last_accessed = int(row[0]), str(row[1])
                        else:
                            # Legacy stores: count via the primary-key index,
                            # avoid MAX(timestamp) on rows containing large BLOBs.
                            frame_count = int(connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0])
                except (OSError, sqlite3.DatabaseError):
                    pass
            entries.append(
                {
                    "path": str(path),
                    "bytes": size,
                    "frames": frame_count,
                    "last_accessed_at": last_accessed,
                }
            )
    return {
        "cache_root": str(root),
        "frames_root": str(frames_root),
        "bytes": total_bytes,
        "stores": len(entries),
        "frames": sum(int(entry["frames"]) for entry in entries),
        "entries": entries,
    }


def clear_frame_cache(cache_root: str | Path) -> dict[str, int]:
    """Clear frame stores and their workspace index while preserving other caches."""
    root = Path(cache_root).resolve()
    frames_root = (root / "frames").resolve()
    if root not in frames_root.parents:
        raise ValueError("Frame cache path must remain inside the cache root.")
    before = inspect_frame_cache(root)
    if frames_root.exists():
        shutil.rmtree(frames_root)
    index_path = root / "index" / "frames.json"
    index_path.unlink(missing_ok=True)
    return {"bytes": int(before["bytes"]), "stores": int(before["stores"])}


def _prune_workspace_index(cache_root: Path) -> None:
    index_path = cache_root / "index" / "frames.json"
    if not index_path.exists():
        return
    with _INDEX_LOCK:
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        entries = payload.get("entries") or {}
        payload["entries"] = {
            key: value
            for key, value in entries.items()
            if Path(str(value.get("path") or "")).exists()
        }
        tmp = index_path.with_name(f"{index_path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(index_path)
        finally:
            tmp.unlink(missing_ok=True)


def enforce_frame_cache_limit(
        cache_root: str | Path,
        max_bytes: int,
        *,
        exclude_generations: Iterable[str | Path] = (),
) -> dict[str, int]:
    """Evict least-recently-used source generations until ``max_bytes`` is met."""
    limit = int(max_bytes)
    if limit <= 0:
        return {"bytes_before": 0, "bytes_after": 0, "evicted_generations": 0}
    root = Path(cache_root).resolve()
    frames_root = root / "frames"
    info = inspect_frame_cache(root, details=False)
    before = int(info["bytes"])
    if before <= limit:
        return {"bytes_before": before, "bytes_after": before, "evicted_generations": 0}
    excluded = {Path(item).resolve() for item in exclude_generations}
    generations: dict[Path, dict[str, Any]] = {}
    for entry in info["entries"]:
        db_path = Path(entry["path"])
        generation = db_path.parent.resolve()
        record = generations.setdefault(generation, {"bytes": 0, "last": "", "dbs": []})
        record["bytes"] += int(entry["bytes"])
        record["last"] = max(str(record["last"]), str(entry["last_accessed_at"]))
        record["dbs"].append(db_path)
    ordered = sorted(
        generations.items(),
        key=lambda item: (str(item[1]["last"]), str(item[0])),
    )
    remaining = before
    evicted = 0
    for generation, record in ordered:
        if remaining <= limit:
            break
        if generation in excluded or frames_root.resolve() not in generation.parents:
            continue
        try:
            with _generation_guard(root, generation, exclusive=True, timeout=0):
                # Hold the lease through deletion; a new stream cannot start
                # between a lock probe and rmtree.
                shutil.rmtree(generation)
        except (OSError, sqlite3.DatabaseError):
            continue
        remaining -= int(record["bytes"])
        evicted += 1
    _prune_workspace_index(root)
    return {
        "bytes_before": before,
        "bytes_after": max(0, remaining),
        "evicted_generations": evicted,
    }


__all__ = [
    "DEFAULT_BUSY_TIMEOUT_SECONDS",
    "DEFAULT_FRAME_CACHE_MAX_BYTES",
    "DEFAULT_SIGNATURE_BLOCK_SIZE",
    "FRAME_PAYLOAD_VERSION",
    "FRAME_STORE_SCHEMA_VERSION",
    "FrameOffset",
    "FrameSourceIdentity",
    "FrameStore",
    "FrameStoreError",
    "FrameStoreIdentityError",
    "FrameStoreVersionError",
    "FrameViewIdentity",
    "IndexCoverage",
    "InputCachePolicy",
    "clear_frame_cache",
    "enforce_frame_cache_limit",
    "inspect_frame_cache",
]
