"""Lightweight caching utilities for saving and loading ReaxKit objects."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import pickle

@dataclass
class CacheConfig:
    """Configuration for cache saving/loading."""
    protocol: int = pickle.HIGHEST_PROTOCOL
    compress: bool = False  # placeholder if you later want compression

def save_cache_blob(path: Path, obj: Any, *, cfg: Optional[CacheConfig] = None) -> None:
    """Save object to pickle cache."""
    cfg = cfg or CacheConfig()
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=cfg.protocol)

def load_cache_blob(path: Path, *, cfg: Optional[CacheConfig] = None) -> Any:
    """Load object from pickle cache."""
    cfg = cfg or CacheConfig()
    with open(path, "rb") as f:
        return pickle.load(f)
