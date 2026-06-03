"""
Base file-handler abstraction for ReaxKit.

This module defines the abstract ``FileHandler`` class, which provides the
common interface and lifecycle used by all ReaxKit file handlers
(e.g., ``XmoloutHandler``, ``Fort7Handler``, ``SummaryHandler``).

The base class standardizes how ReaxFF output files are:

- loaded from disk
- parsed lazily into structured tabular data
- exposed via a uniform DataFrame-based API
- accompanied by lightweight metadata

All ReaxKit analysis functions rely on ``FileHandler`` subclasses to provide
a consistent, predictable view of parsed ReaxFF files.

**Usage context**

- ReaxFF parsing: Read ReaxFF text outputs into normalized tabular structures.
- Workflow ingestion: Provide canonical handler interfaces used by adapters/workflows.
- Diagnostics/export: Preserve parsed metadata for reporting and downstream conversion.
"""


from __future__ import annotations
from abc import ABC, abstractmethod
from hashlib import sha256
from pathlib import Path
from typing import Any
import os
import pickle
import threading
import json
from datetime import datetime, timezone
import pandas as pd
from reaxkit.core.platform.exceptions import ParseError

try:  # pragma: no cover - optional at runtime
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None  # type: ignore

class BaseHandler(ABC):
    """
    Abstract base class for ReaxKit file handlers.

    This class defines the minimal public interface that all ReaxKit
    file handlers must implement. Subclasses are responsible for parsing
    a specific ReaxFF file format and exposing its contents as structured
    pandas DataFrames.

    Parsed Data
    -----------
    Main table
        A pandas.DataFrame returned by ``dataframe()``, whose columns
        depend on the specific file type.

    Metadata
        A dictionary of lightweight metadata returned by ``metadata()``,
        typically including global or per-file attributes.

    Notes
    -----
    - Parsing is performed lazily and cached after the first access.
    - Parsed payloads are also cached across handler instances using:
      1) an in-process memory cache
      2) an on-disk cache
    - Subclasses must implement the private ``_parse()`` method.
    """

    _CACHE_ENV_VAR = "REAXKIT_HANDLER_CACHE_DIR"
    _CACHE_VERSION = "2"
    _MEMORY_CACHE: dict[str, bytes] = {}
    _MEMORY_LOCK = threading.Lock()

    def __init__(self, file_path: str | Path):
        """
        Initialize a file handler with a file path.

        Works on
        --------
        ReaxFF output files on disk

        Parameters
        ----------
        file_path : str or pathlib.Path
            Path to the file to be parsed.

        Returns
        -------
        None
            Initializes the handler without parsing the file.
        """
        self.path = Path(file_path)
        self._parsed = False
        self._df: pd.DataFrame | None = None
        self._meta: dict[str, Any] = {}

    # ---- public API
    def parse(self) -> None:
        """
            Parse the file contents into structured data.

            Works on
            --------
            FileHandler — ReaxFF output file

            Returns
            -------
            None
                Parses the file and caches the resulting DataFrame and metadata.

            Examples
            --------
            >>> h = SomeHandler("file")
            >>> h.parse()
            """
        if not self._parsed:
            handler_id = self._handler_id()

            payload = self._load_from_memory_cache(handler_id)
            if payload is None:
                payload = self._load_from_disk_cache(handler_id)
                if payload is not None:
                    self._store_in_memory_cache(handler_id, payload)

            if payload is not None:
                self._restore_cached_payload(payload)
                return

            try:
                df, meta = self._parse()
            except ParseError:
                raise
            except Exception as exc:
                raise ParseError(
                    f"Failed to parse {self.__class__.__name__} file at '{self.path}': {exc}"
                ) from exc
            self._df = df
            self._meta = meta or {}
            self._parsed = True

            payload = self._build_cached_payload()
            if payload is not None:
                self._store_in_memory_cache(handler_id, payload)
                self._store_in_disk_cache(handler_id, payload)

    def dataframe(self) -> pd.DataFrame:
        """
            Return the parsed file contents as a pandas DataFrame.

            Works on
            --------
            FileHandler — ReaxFF output file

            Returns
            -------
            pandas.DataFrame
                Structured table representing the parsed file contents.

            Examples
            --------
            >>> h = SomeHandler("file")
            >>> df = h.dataframe()
            """
        if not self._parsed:
            self.parse()
        assert self._df is not None
        return self._df

    def metadata(self) -> dict[str, Any]:
        """
            Return parsed metadata associated with the file.

            Works on
            --------
            FileHandler — ReaxFF output file

            Returns
            -------
            dict
                Dictionary of metadata values extracted during parsing.

            Examples
            --------
            >>> h = SomeHandler("file")
            >>> meta = h.metadata()
            """
        if not self._parsed:
            self.parse()
        return dict(self._meta)

    # ---- subclasses must implement
    @abstractmethod
    def _parse(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Parse the file and return structured data and metadata.

        Works on
        --------
        ReaxFF output files

        Parameters
        ----------
        None

        Returns
        -------
        tuple (pandas.DataFrame, dict)
            Parsed data table and associated metadata.

        Notes
        -----
        This method must be implemented by all subclasses."""
        ...

    # ---- shared cache helpers
    @classmethod
    def clear_runtime_cache(cls) -> None:
        """Clear runtime cache.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Return value.

        Examples
        --------
        ```python
        # Example
        clear_runtime_cache(...)
        ```
        """
        with cls._MEMORY_LOCK:
            cls._MEMORY_CACHE.clear()

    def _handler_id(self) -> str:
        """Handler id."""
        identity: dict[str, Any] = {
            "handler": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "cache_version": str(self._CACHE_VERSION),
            "options": self._identity_options(),
        }
        try:
            resolved = self.path.resolve()
            identity["source"] = self._file_fingerprint(resolved)
        except OSError:
            identity["source"] = {"path": str(self.path), "missing": True}
        blob = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return sha256(blob.encode("utf-8")).hexdigest()

    def _identity_options(self) -> dict[str, Any]:
        """Return deterministic handler options that affect parse output."""
        state = self._state_snapshot()
        options: dict[str, Any] = {}
        for name, value in state.items():
            if name.startswith("_report"):
                continue
            options[name] = self._normalize_identity_value(value)
        return options

    @classmethod
    def _normalize_identity_value(cls, value: Any) -> Any:
        """Normalize identity value."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {
                str(k): cls._normalize_identity_value(v)
                for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            }
        if isinstance(value, (list, tuple)):
            return [cls._normalize_identity_value(v) for v in value]
        if isinstance(value, set):
            return sorted(cls._normalize_identity_value(v) for v in value)
        return repr(value)

    @staticmethod
    def _file_fingerprint(path: Path) -> dict[str, Any]:
        """File fingerprint."""
        stat = path.stat()
        digest = sha256()
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return {
            "sha256": digest.hexdigest(),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    # Backward-compatible alias
    def _cache_key(self) -> str:
        """Cache key."""
        return self._handler_id()

    def _cache_root(self) -> Path:
        """Cache root."""
        env_root = os.environ.get(self._CACHE_ENV_VAR, "").strip()
        if env_root:
            return Path(env_root)
        project_workspace = Path("reaxkit_workspace")
        if project_workspace.exists() and project_workspace.is_dir():
            return project_workspace / "cache" / "handlers"
        return Path(".reaxkit_cache") / "handlers"

    def _cache_file_path(self, key: str) -> Path:
        """Cache file path."""
        return self._cache_root() / f"{key}.h5"

    def _disk_cache_dir(self, key: str) -> Path:
        """Disk cache dir."""
        return self._cache_root() / key

    def _disk_cache_h5_path(self, key: str) -> Path:
        """Disk cache h5 path."""
        return self._disk_cache_dir(key) / "cache.h5"

    def _cache_index_path(self) -> Path:
        """Cache index path."""
        return self._cache_root().parent / "index" / "handlers.json"

    @staticmethod
    def _utc_now_iso() -> str:
        """Utc now iso."""
        return datetime.now(timezone.utc).isoformat()

    def _update_cache_index(self, key: str) -> None:
        """Update cache index."""
        index_path = self._cache_index_path()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {}
        if index_path.exists():
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        payload.setdefault("namespace", "handlers")
        payload.setdefault("entries", {})
        payload["entries"][str(key)] = {
            "path": str(self._disk_cache_h5_path(key)),
            "handler": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "updated_at": self._utc_now_iso(),
        }
        index_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _state_snapshot(self) -> dict[str, Any]:
        """State snapshot."""
        excluded = {"path", "_parsed", "_df", "_meta"}
        snapshot: dict[str, Any] = {}
        for name, value in self.__dict__.items():
            if name in excluded:
                continue
            if callable(value):
                continue
            snapshot[name] = value
        return snapshot

    def _build_cached_payload(self) -> bytes | None:
        """Build cached payload."""
        if self._df is None:
            return None
        payload = {"df": self._df, "meta": dict(self._meta), "state": self._state_snapshot()}
        try:
            return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            return None

    def _restore_cached_payload(self, payload: bytes) -> None:
        """Restore cached payload."""
        obj = pickle.loads(payload)
        self._df = obj.get("df")
        self._meta = dict(obj.get("meta") or {})
        state = obj.get("state") or {}
        for name, value in state.items():
            self.__dict__[name] = value
        self._parsed = True

    @classmethod
    def _load_from_memory_cache(cls, key: str) -> bytes | None:
        """Load from memory cache."""
        with cls._MEMORY_LOCK:
            return cls._MEMORY_CACHE.get(key)

    @classmethod
    def _store_in_memory_cache(cls, key: str, payload: bytes) -> None:
        """Store in memory cache."""
        with cls._MEMORY_LOCK:
            cls._MEMORY_CACHE[key] = payload

    def _load_from_disk_cache(self, key: str) -> bytes | None:
        """Load from disk cache."""
        if h5py is None:
            return None
        path = self._disk_cache_h5_path(key)
        if not path.exists():
            return None
        try:
            with h5py.File(path, "r") as h5:
                if "payload" not in h5:
                    return None
                arr = h5["payload"][...]
            return bytes(bytearray(arr.tolist()))
        except Exception:
            return None

    def _store_in_disk_cache(self, key: str, payload: bytes) -> None:
        """Store in disk cache."""
        if h5py is None:
            return
        tmp_dir: Path | None = None
        cache_dir = self._disk_cache_dir(key)
        try:
            tmp_dir = cache_dir.with_name(f"{cache_dir.name}.__tmpdir__")
            if tmp_dir.exists():
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            h5_path = tmp_dir / "cache.h5"
            with h5py.File(h5_path, "w") as h5:
                h5.attrs["cache_format"] = "reaxkit-handler-hdf5-v1"
                h5.attrs["handler"] = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
                h5.create_dataset("payload", data=list(payload), dtype="u1")

            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir, ignore_errors=True)
            tmp_dir.replace(cache_dir)
            self._update_cache_index(key)
            return
        except Exception:
            if tmp_dir is not None and tmp_dir.exists():
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
            return
