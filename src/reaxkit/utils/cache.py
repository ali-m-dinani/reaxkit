"""
Lightweight caching utilities for ReaxKit objects.

This module provides minimal helpers for serializing and deserializing
Python objects to disk using pickle. It is intended for caching intermediate
results such as parsed handlers, analysis outputs, or precomputed tables
to avoid repeated expensive computations.

The API is intentionally small and explicit to keep caching behavior
predictable and easy to reason about.
"""


from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import pickle

@dataclass
class CacheConfig:
    """
    Configuration options for cache serialization.

    Parameters
    ----------
    protocol : int
        Pickle protocol version to use when saving objects.
    compress : bool
        Placeholder flag for future compression support.
        """
    protocol: int = pickle.HIGHEST_PROTOCOL
    compress: bool = False  # placeholder if you later want compression

def save_cache_blob(path: Path, obj: Any, *, cfg: Optional[CacheConfig] = None) -> None:
    """
    Save a Python object to a cache file.

    The object is serialized using pickle and written to the specified
    path. Existing files will be overwritten.

    Parameters
    ----------
    path : pathlib.Path
        Destination path for the cache file.
    obj : Any
        Python object to serialize.
    cfg : CacheConfig, optional
        Cache configuration controlling serialization behavior.

    Returns
    -------
    None
    """
    cfg = cfg or CacheConfig()
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=cfg.protocol)

def load_cache_blob(path: Path, *, cfg: Optional[CacheConfig] = None) -> Any:
    """
    Load a Python object from a cache file.

    Parameters
    ----------
    path : pathlib.Path
        Path to the cache file.
    cfg : CacheConfig, optional
        Cache configuration (reserved for future extensions).

    Returns
    -------
    Any
        Deserialized Python object stored in the cache.
        """
    cfg = cfg or CacheConfig()
    with open(path, "rb") as f:
        return pickle.load(f)
