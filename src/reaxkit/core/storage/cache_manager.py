"""Core-level cache management for analysis executor."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional
import pickle
import threading
import numpy as np
import json
from datetime import datetime, timezone
from dataclasses import asdict, is_dataclass

try:  # pragma: no cover - optional at runtime
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None  # type: ignore


@dataclass
class CacheConfig:
    """Configuration for on-disk cache behavior."""

    root: Path = Path(".reaxkit_cache")
    protocol: int = pickle.HIGHEST_PROTOCOL
    namespace: str = "analysis"
    memory_maxsize: int = 128


class CacheManager:
    """Two-layer cache for task results (in-memory LRU + file-backed)."""

    _MEMORY_POOLS: dict[tuple[str, str], OrderedDict[str, Any]] = {}
    _MEMORY_LOCK = threading.Lock()

    def __init__(self, cfg: Optional[CacheConfig] = None):
        self.cfg = cfg or CacheConfig()
        self._root = Path(self.cfg.root) / self.cfg.namespace
        self._root.mkdir(parents=True, exist_ok=True)
        self._pool_key = (str(self._root.resolve()), self.cfg.namespace)

    def _memory_pool_locked(self) -> OrderedDict[str, Any]:
        pool = self._MEMORY_POOLS.get(self._pool_key)
        if pool is None:
            pool = OrderedDict()
            self._MEMORY_POOLS[self._pool_key] = pool
        return pool

    def _memory_get(self, key: str) -> tuple[bool, Any]:
        with self._MEMORY_LOCK:
            pool = self._memory_pool_locked()
            if key not in pool:
                return False, None
            value = pool.pop(key)
            pool[key] = value
            return True, value

    def _memory_put(self, key: str, value: Any) -> None:
        if int(self.cfg.memory_maxsize) <= 0:
            return
        with self._MEMORY_LOCK:
            pool = self._memory_pool_locked()
            if key in pool:
                pool.pop(key)
            pool[key] = value
            while len(pool) > int(self.cfg.memory_maxsize):
                pool.popitem(last=False)

    @classmethod
    def clear_memory_cache(cls) -> None:
        with cls._MEMORY_LOCK:
            cls._MEMORY_POOLS.clear()

    def _blob(self, obj: Any) -> bytes:
        try:
            return pickle.dumps(obj, protocol=self.cfg.protocol)
        except Exception:
            return repr(obj).encode("utf-8", errors="replace")

    def _normalize_for_key(self, obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if is_dataclass(obj):
            return {str(k): self._normalize_for_key(v) for k, v in asdict(obj).items()}
        if isinstance(obj, dict):
            return {str(k): self._normalize_for_key(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
        if isinstance(obj, (list, tuple)):
            return [self._normalize_for_key(v) for v in obj]
        if isinstance(obj, set):
            return sorted(self._normalize_for_key(v) for v in obj)
        return repr(obj)

    def _request_fingerprint(self, request: Any) -> str:
        normalized = self._normalize_for_key(request)
        blob = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return sha256(blob.encode("utf-8")).hexdigest()

    def analysis_id_for(
        self,
        *,
        task: Any,
        data: Any,
        request: Any,
        parsed_id: str | None = None,
        task_version: str = "1",
    ) -> str:
        h = sha256()
        task_name = getattr(task, "__class__", type(task)).__name__
        if parsed_id:
            h.update(str(parsed_id).encode("utf-8"))
            h.update(b"|")
            h.update(str(task_name).encode("utf-8"))
            h.update(b"|")
            h.update(str(task_version).encode("utf-8"))
            h.update(b"|")
            h.update(self._request_fingerprint(request).encode("utf-8"))
            return h.hexdigest()

        h.update(str(task_name).encode("utf-8"))
        h.update(self._blob(data))
        h.update(self._blob(request))
        return h.hexdigest()

    # Backward-compatible alias
    def key_for(
        self,
        *,
        task: Any,
        data: Any,
        request: Any,
        parsed_id: str | None = None,
        task_version: str = "1",
    ) -> str:
        return self.analysis_id_for(
            task=task,
            data=data,
            request=request,
            parsed_id=parsed_id,
            task_version=task_version,
        )

    def _path(self, analysis_id: str) -> Path:
        return self._root / str(analysis_id) / "cache.h5"

    def _index_path(self) -> Path:
        return Path(self.cfg.root) / "index" / f"{self.cfg.namespace}.json"

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _update_index(self, analysis_id: str, *, cache_path: Path, task_name: str | None = None) -> None:
        path = self._index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {}
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        payload.setdefault("namespace", self.cfg.namespace)
        payload.setdefault("entries", {})
        entry: dict[str, Any] = {
            "path": str(cache_path),
            "updated_at": self._utc_now_iso(),
        }
        if task_name:
            entry["task_name"] = str(task_name)
        payload["entries"][str(analysis_id)] = entry
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def exists(self, analysis_id: str) -> bool:
        in_mem, _ = self._memory_get(analysis_id)
        if in_mem:
            return True
        return self._path(analysis_id).exists()

    def load(self, analysis_id: str) -> Any:
        in_mem, value = self._memory_get(analysis_id)
        if in_mem:
            return value
        if h5py is None:
            raise RuntimeError("h5py is required for on-disk analysis cache.")
        with h5py.File(self._path(analysis_id), "r") as h5:
            if "payload" not in h5:
                raise KeyError(f"Missing payload dataset for cache analysis_id={analysis_id}.")
            payload = bytes(bytearray(h5["payload"][...].tolist()))
        value = pickle.loads(payload)
        self._memory_put(analysis_id, value)
        return value

    def store(self, analysis_id: str, value: Any, *, task_name: str | None = None) -> None:
        self._memory_put(analysis_id, value)
        if h5py is None:
            return
        path = self._path(analysis_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = pickle.dumps(value, protocol=self.cfg.protocol)
        tmp = path.with_suffix(".tmp")
        with h5py.File(tmp, "w") as h5:
            h5.attrs["cache_format"] = "reaxkit-analysis-hdf5-v1"
            h5.create_dataset("payload", data=np.frombuffer(payload, dtype=np.uint8))
        tmp.replace(path)
        self._update_index(analysis_id, cache_path=path, task_name=task_name)


__all__ = ["CacheConfig", "CacheManager"]
