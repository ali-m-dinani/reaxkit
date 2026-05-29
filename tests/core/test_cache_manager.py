from __future__ import annotations

from pathlib import Path

import pytest

from reaxkit.core.storage.cache_manager import CacheConfig, CacheManager


def _new_cache(root: Path, *, memory_maxsize: int = 128) -> CacheManager:
    return CacheManager(
        CacheConfig(
            root=root,
            namespace="analysis",
            memory_maxsize=memory_maxsize,
        )
    )


def test_cachemanager_memory_hit_across_instances(tmp_path: Path):
    CacheManager.clear_memory_cache()
    cache = _new_cache(tmp_path, memory_maxsize=8)
    cache.store("k1", {"value": 11})

    # Remove disk blob to prove memory layer serves across instances.
    disk_path = tmp_path / "analysis" / "k1.pkl"
    assert disk_path.exists()
    disk_path.unlink()

    cache2 = _new_cache(tmp_path, memory_maxsize=8)
    assert cache2.exists("k1")
    assert cache2.load("k1") == {"value": 11}


def test_cachemanager_load_promotes_disk_value_into_memory(tmp_path: Path):
    CacheManager.clear_memory_cache()
    cache = _new_cache(tmp_path, memory_maxsize=8)
    cache.store("k2", [1, 2, 3])

    CacheManager.clear_memory_cache()
    cache2 = _new_cache(tmp_path, memory_maxsize=8)
    assert cache2.load("k2") == [1, 2, 3]

    # After disk load, memory should now have it; deleting disk should still allow load.
    disk_path = tmp_path / "analysis" / "k2.pkl"
    disk_path.unlink()
    cache3 = _new_cache(tmp_path, memory_maxsize=8)
    assert cache3.load("k2") == [1, 2, 3]


def test_cachemanager_memory_lru_eviction(tmp_path: Path):
    CacheManager.clear_memory_cache()
    cache = _new_cache(tmp_path, memory_maxsize=2)
    cache.store("a", 1)
    cache.store("b", 2)
    assert cache.load("a") == 1  # make "a" most recently used
    cache.store("c", 3)          # evicts "b"

    disk_dir = tmp_path / "analysis"
    for p in disk_dir.glob("*.pkl"):
        p.unlink()

    cache2 = _new_cache(tmp_path, memory_maxsize=2)
    assert cache2.load("a") == 1
    assert cache2.load("c") == 3
    with pytest.raises(FileNotFoundError):
        cache2.load("b")

