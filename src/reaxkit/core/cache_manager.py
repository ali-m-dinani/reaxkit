"""Core-level cache management for analysis executor."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional
import pickle


@dataclass
class CacheConfig:
    """Configuration for on-disk cache behavior."""

    root: Path = Path(".reaxkit_cache")
    protocol: int = pickle.HIGHEST_PROTOCOL
    namespace: str = "analysis"


class CacheManager:
    """Simple file-backed cache for task results."""

    def __init__(self, cfg: Optional[CacheConfig] = None):
        self.cfg = cfg or CacheConfig()
        self._root = Path(self.cfg.root) / self.cfg.namespace
        self._root.mkdir(parents=True, exist_ok=True)

    def _blob(self, obj: Any) -> bytes:
        try:
            return pickle.dumps(obj, protocol=self.cfg.protocol)
        except Exception:
            return repr(obj).encode("utf-8", errors="replace")

    def key_for(self, *, task: Any, data: Any, request: Any) -> str:
        h = sha256()
        h.update(getattr(task, "__class__", type(task)).__name__.encode("utf-8"))
        h.update(self._blob(data))
        h.update(self._blob(request))
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        return self._root / f"{key}.pkl"

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def load(self, key: str) -> Any:
        with open(self._path(key), "rb") as f:
            return pickle.load(f)

    def store(self, key: str, value: Any) -> None:
        path = self._path(key)
        with open(path, "wb") as f:
            pickle.dump(value, f, protocol=self.cfg.protocol)


__all__ = ["CacheConfig", "CacheManager"]
