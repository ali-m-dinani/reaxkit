"""Load AMS ``.kf/.rkf`` files with process-local handle caching.

This module centralizes lazy ``KFFile`` construction and cache-key generation
for AMS-backed engine loaders. It is intentionally small and only manages
handle lifecycle concerns (existence checks, optional load timing callback,
and in-memory reuse keyed by file identity).

**Usage context**

- Adapter internals: Reused by ``AMSAdapter`` loaders to avoid repeated opens.
- Runtime performance: Skips expensive re-loads for unchanged KF/RKF files.
- Diagnostics: Emits optional per-load timing via callback when uncached.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Callable
import threading


class RKFHandler:
    """Cached loader for AMS ``KFFile`` handles."""

    _CACHE_VERSION = "1"
    _MEMORY_CACHE: dict[str, Any] = {}
    _MEMORY_LOCK = threading.Lock()

    def __init__(self, file_path: str | Path, timing_callback: Callable[[Path, float], None] | None = None):
        """Initialize a handler bound to one AMS KF/RKF path."""
        self.path = Path(file_path)
        self._timing_callback = timing_callback

    @classmethod
    def clear_runtime_cache(cls) -> None:
        """Clear all cached ``KFFile`` handles for the current Python process.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Examples
        --------
        ```python
        RKFHandler.clear_runtime_cache()
        ```
        """
        with cls._MEMORY_LOCK:
            cls._MEMORY_CACHE.clear()

    @classmethod
    def _cache_key(cls, path: Path) -> str:
        """Build a stable cache key from resolved path and file metadata."""
        try:
            resolved = path.resolve()
            stat = resolved.stat()
            identity = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}|{cls._CACHE_VERSION}"
        except OSError:
            identity = f"{path}|missing|{cls._CACHE_VERSION}"
        return sha256(identity.encode("utf-8")).hexdigest()

    @staticmethod
    def _load_uncached(path: Path):
        """Construct a new AMS ``KFFile`` handle without cache lookup."""
        try:
            from scm.plams.tools.kftools import KFFile
        except Exception as exc:  # pragma: no cover - optional dependency at runtime
            raise RuntimeError(
                "AMS loading requires 'scm.plams.tools.kftools.KFFile'. "
                "Install AMS/PLAMS Python modules and retry."
            ) from exc
        return KFFile(str(path))

    def kf(self):
        """Return a cached AMS ``KFFile`` handle for this path.

        Loads on first access and then reuses the same in-memory handle for
        subsequent calls while the underlying file identity is unchanged.

        Parameters
        ----------
        None

        Returns
        -------
        Any
            ``scm.plams.tools.kftools.KFFile`` instance bound to ``self.path``.

        Examples
        --------
        ```python
        handler = RKFHandler("reaxout.kf")
        kf = handler.kf()
        ```
        """
        if not self.path.exists():
            raise FileNotFoundError(f"AMS KF/RKF file not found: {self.path}")

        key = self._cache_key(self.path)
        with self._MEMORY_LOCK:
            cached = self._MEMORY_CACHE.get(key)
        if cached is not None:
            return cached

        t0 = perf_counter()
        kf = self._load_uncached(self.path)
        if callable(self._timing_callback):
            self._timing_callback(self.path, perf_counter() - t0)

        with self._MEMORY_LOCK:
            self._MEMORY_CACHE[key] = kf
        return kf
