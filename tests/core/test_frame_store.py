from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import threading

import pytest

from reaxkit.core.storage.frame_store import (
    FRAME_STORE_SCHEMA_VERSION,
    FrameOffset,
    FrameSourceIdentity,
    FrameStore,
    FrameStoreVersionError,
    FrameViewIdentity,
    IndexCoverage,
    InputCachePolicy,
    clear_frame_cache,
    enforce_frame_cache_limit,
    inspect_frame_cache,
)


def _source(tmp_path: Path, content: bytes = b"frame source\n") -> Path:
    path = tmp_path / "xmolout"
    path.write_bytes(content)
    return path


def _store(
        tmp_path: Path,
        source_path: Path,
        *,
        parser_version: str = "1",
        options: dict | None = None,
) -> FrameStore:
    return FrameStore.for_source(
        tmp_path / "cache",
        source_path,
        engine="reaxff",
        source_kind="xmolout",
        parser="XmoloutHandler",
        parser_version=parser_version,
        representation="atom-frame-v1",
        capabilities=("coordinates", "atom-labels", "cell"),
        options=options or {"extra_atom_cols": None},
    )


def test_source_and_view_identities_are_deterministic_and_versioned(tmp_path: Path):
    source_path = _source(tmp_path)
    first = FrameSourceIdentity.from_path(
        source_path,
        engine="ReaxFF",
        source_kind="xmolout",
    )
    second = FrameSourceIdentity.from_path(
        source_path,
        engine="reaxff",
        source_kind="xmolout",
    )

    assert first.source_key == second.source_key
    assert first.generation_key == second.generation_key

    view_a = FrameViewIdentity(
        parser="XmoloutHandler",
        parser_version="1",
        representation="atom-frame-v1",
        capabilities=("cell", "coordinates", "coordinates"),
        options={"columns": ["x", "y", "z"], "strict": True},
    )
    view_b = FrameViewIdentity(
        parser="XmoloutHandler",
        parser_version="1",
        representation="atom-frame-v1",
        capabilities=("coordinates", "cell"),
        options={"strict": True, "columns": ["x", "y", "z"]},
    )
    view_new_parser = FrameViewIdentity(
        parser="XmoloutHandler",
        parser_version="2",
        representation="atom-frame-v1",
        capabilities=("coordinates", "cell"),
        options={"strict": True, "columns": ["x", "y", "z"]},
    )

    assert view_a.view_key == view_b.view_key
    assert view_a.view_key != view_new_parser.view_key
    assert view_a.view_key != FrameViewIdentity(
        parser="XmoloutHandler",
        parser_version="1",
        representation="atom-frame-v1",
        capabilities=("coordinates", "cell"),
        options={"strict": False, "columns": ["x", "y", "z"]},
    ).view_key
    assert view_a.can_satisfy(capabilities=("coordinates",))
    assert not view_a.can_satisfy(capabilities=("charges",))


def test_source_mutation_creates_a_new_generation(tmp_path: Path):
    source_path = _source(tmp_path, b"alpha\n")
    original_stat = source_path.stat()
    before = FrameSourceIdentity.from_path(
        source_path,
        engine="reaxff",
        source_kind="xmolout",
    )

    source_path.write_bytes(b"bravo\n")
    os.utime(
        source_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    after = FrameSourceIdentity.from_path(
        source_path,
        engine="reaxff",
        source_kind="xmolout",
    )

    assert before.source_key == after.source_key
    assert before.generation_key != after.generation_key


def test_frame_store_persists_frames_offsets_coverage_and_index(tmp_path: Path):
    source_path = _source(tmp_path)
    first = _store(tmp_path, source_path)

    assert first.put_frames(
        {
            0: {"iteration": 0, "coordinates": [[0.0, 0.0, 0.0]]},
            200: {"iteration": 2000, "coordinates": [[1.0, 0.0, 0.0]]},
        }
    ) == {0, 200}
    assert first.put_offsets(
        [
            FrameOffset(0, 0, 100, iteration=0, atom_count=1),
            FrameOffset(200, 1000, 1100, iteration=2000, atom_count=1),
        ]
    ) == {0, 200}
    assert first.set_coverage(IndexCoverage(201, 1100, complete=False))

    second = _store(tmp_path, source_path)
    assert list(second.get_frames([200, 0, 40])) == [200, 0]
    assert second.get_frames([0])[0]["iteration"] == 0
    assert second.missing_indices([0, 40, 200]) == (40,)
    assert second.get_offsets([200, 40]) == {
        200: FrameOffset(200, 1000, 1100, iteration=2000, atom_count=1)
    }
    assert second.get_coverage() == IndexCoverage(201, 1100, complete=False)
    assert not second.set_coverage(IndexCoverage(200, 1099, complete=False))
    assert second.get_coverage() == IndexCoverage(201, 1100, complete=False)

    index = json.loads(second.index_path.read_text(encoding="utf-8"))
    assert index["namespace"] == "frames"
    assert any(entry["path"] == str(second.path) for entry in index["entries"].values())


def test_corrupt_frame_payload_is_removed_and_becomes_a_cache_miss(tmp_path: Path):
    store = _store(tmp_path, _source(tmp_path))
    store.put_frames({4: {"value": "valid"}})

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE frames SET payload = ? WHERE frame_index = ?",
            (b"corrupt", 4),
        )

    assert store.get_frames([4]) == {}
    assert store.available_indices([4]) == set()


def test_incompatible_schema_version_is_rejected(tmp_path: Path):
    source_path = _source(tmp_path)
    store = _store(tmp_path, source_path)
    with sqlite3.connect(store.path) as connection:
        connection.execute(f"PRAGMA user_version = {FRAME_STORE_SCHEMA_VERSION + 100}")

    with pytest.raises(FrameStoreVersionError, match="Unsupported frame-store schema"):
        _store(tmp_path, source_path)


def test_concurrent_writers_produce_complete_readable_store(tmp_path: Path):
    store = _store(tmp_path, _source(tmp_path), options={"mode": "concurrent"})

    def write(frame_index: int) -> set[int]:
        peer = FrameStore(store.cache_root, source=store.source, view=store.view)
        return peer.put_frames({frame_index: {"frame_index": frame_index}})

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(write, range(32)))

    assert all(result for result in results)
    loaded = store.get_frames(range(32))
    assert set(loaded) == set(range(32))
    assert all(loaded[index]["frame_index"] == index for index in range(32))


def test_reader_can_use_store_while_writer_adds_frames(tmp_path: Path):
    store = _store(tmp_path, _source(tmp_path), options={"mode": "read-write"})
    barrier = threading.Barrier(2)

    def write_frames() -> None:
        peer = FrameStore(store.cache_root, source=store.source, view=store.view)
        barrier.wait()
        for frame_index in range(24):
            assert peer.put_frames({frame_index: {"frame_index": frame_index}})

    def read_frames() -> None:
        peer = FrameStore(store.cache_root, source=store.source, view=store.view)
        barrier.wait()
        for _ in range(24):
            for frame_index, payload in peer.get_frames(range(24)).items():
                assert payload["frame_index"] == frame_index

    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(write_frames)
        reader = pool.submit(read_frames)
        writer.result()
        reader.result()

    assert set(store.get_frames(range(24))) == set(range(24))


def test_interrupted_frame_transaction_does_not_expose_partial_writes(
        tmp_path: Path,
        monkeypatch,
):
    store = _store(tmp_path, _source(tmp_path), options={"mode": "interrupted"})
    original_connect = store._connect

    class _FailingConnection:
        def __init__(self):
            self.connection = original_connect()

        def __enter__(self):
            self.connection.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return self.connection.__exit__(exc_type, exc_value, traceback)

        def execute(self, *args, **kwargs):
            return self.connection.execute(*args, **kwargs)

        def executemany(self, statement, rows):
            first = next(iter(rows))
            self.connection.execute(statement, first)
            raise sqlite3.OperationalError("simulated interrupted write")

        def close(self):
            self.connection.close()

    with monkeypatch.context() as patch:
        patch.setattr(store, "_connect", _FailingConnection)
        assert store.put_frames({10: {"value": 10}, 20: {"value": 20}}) == set()

    assert store.get_frames([10, 20]) == {}


def test_input_cache_policy_is_separate_from_analysis_cache_flags():
    assert InputCachePolicy.from_args({"cache": False}).enabled
    assert InputCachePolicy.from_args({"no_cache": True}).enabled
    assert not InputCachePolicy.from_args({"input_cache": False}).enabled
    assert not InputCachePolicy.from_args({"no_input_cache": True}).enabled


def test_frame_store_has_no_optional_io_dependency(tmp_path: Path):
    store = _store(tmp_path, _source(tmp_path))

    assert store.path.suffix == ".sqlite3"
    assert store.put_frames({0: [1, 2, 3]}) == {0}
    assert store.get_frames([0]) == {0: [1, 2, 3]}


def test_frame_cache_inspection_and_clear_are_index_consistent(tmp_path: Path):
    store = _store(tmp_path, _source(tmp_path))
    store.put_frames({0: {"value": "cached"}})

    info = inspect_frame_cache(store.cache_root)
    assert info["stores"] == 1
    assert info["frames"] == 1
    assert info["bytes"] > 0
    assert store.index_path.exists()

    cleared = clear_frame_cache(store.cache_root)
    assert cleared["stores"] == 1
    assert not (store.cache_root / "frames").exists()
    assert not store.index_path.exists()


def test_frame_cache_limit_evicts_old_generation_and_keeps_active_one(tmp_path: Path):
    first_source = tmp_path / "first.xmolout"
    second_source = tmp_path / "second.xmolout"
    first_source.write_bytes(b"first\n")
    second_source.write_bytes(b"second\n")
    first = _store(tmp_path, first_source, options={"source": "first"})
    second = _store(tmp_path, second_source, options={"source": "second"})
    first.put_frames({0: {"payload": "x" * 4096}})
    second.put_frames({0: {"payload": "y" * 4096}})
    before = inspect_frame_cache(first.cache_root)

    result = enforce_frame_cache_limit(
        first.cache_root,
        max(1, before["bytes"] - 1),
        exclude_generations={second.path.parent},
    )

    assert result["evicted_generations"] == 1
    assert not first.path.parent.exists()
    assert second.path.exists()
