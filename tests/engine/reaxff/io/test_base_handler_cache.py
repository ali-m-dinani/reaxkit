from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from reaxkit.engine.reaxff.io.base import BaseHandler


class _CountingHandler(BaseHandler):
    parse_calls = 0

    def __init__(self, file_path: str | Path):
        super().__init__(file_path)
        self._extra_len = -1

    def _parse(self):
        type(self).parse_calls += 1
        text = self.path.read_text(encoding="utf-8")
        self._extra_len = len(text)
        return pd.DataFrame({"n_chars": [len(text)]}), {"n_chars": len(text)}

    def extra_len(self) -> int:
        if not self._parsed:
            self.parse()
        return int(self._extra_len)


def test_basehandler_runtime_cache_across_instances(tmp_path: Path, monkeypatch):
    cache_root = tmp_path / "handler_cache"
    monkeypatch.setenv(BaseHandler._CACHE_ENV_VAR, str(cache_root))
    BaseHandler.clear_runtime_cache()
    _CountingHandler.parse_calls = 0

    src = tmp_path / "sample.txt"
    src.write_text("alpha\nbeta\n", encoding="utf-8")

    first = _CountingHandler(src)
    second = _CountingHandler(src)

    assert int(first.dataframe().iloc[0]["n_chars"]) == len("alpha\nbeta\n")
    assert int(second.dataframe().iloc[0]["n_chars"]) == len("alpha\nbeta\n")
    assert _CountingHandler.parse_calls == 1


def test_basehandler_disk_cache_after_runtime_clear_restores_state(tmp_path: Path, monkeypatch):
    cache_root = tmp_path / "handler_cache"
    monkeypatch.setenv(BaseHandler._CACHE_ENV_VAR, str(cache_root))
    BaseHandler.clear_runtime_cache()
    _CountingHandler.parse_calls = 0

    src = tmp_path / "sample.txt"
    src.write_text("gamma\ndelta\n", encoding="utf-8")
    expected_len = len("gamma\ndelta\n")

    first = _CountingHandler(src)
    assert int(first.dataframe().iloc[0]["n_chars"]) == expected_len
    assert first.extra_len() == expected_len
    assert _CountingHandler.parse_calls == 1

    BaseHandler.clear_runtime_cache()

    second = _CountingHandler(src)
    assert int(second.dataframe().iloc[0]["n_chars"]) == expected_len
    assert second.extra_len() == expected_len
    assert _CountingHandler.parse_calls == 1

    cache_dirs = [p for p in cache_root.iterdir() if p.is_dir()]
    cache_files = list(cache_root.glob("*.pkl"))
    assert cache_dirs or cache_files


def test_basehandler_cache_invalidation_when_file_changes(tmp_path: Path, monkeypatch):
    cache_root = tmp_path / "handler_cache"
    monkeypatch.setenv(BaseHandler._CACHE_ENV_VAR, str(cache_root))
    BaseHandler.clear_runtime_cache()
    _CountingHandler.parse_calls = 0

    src = tmp_path / "sample.txt"
    src.write_text("one\n", encoding="utf-8")

    first = _CountingHandler(src)
    assert int(first.dataframe().iloc[0]["n_chars"]) == len("one\n")
    assert _CountingHandler.parse_calls == 1

    time.sleep(0.01)
    src.write_text("one\ntwo\n", encoding="utf-8")

    second = _CountingHandler(src)
    assert int(second.dataframe().iloc[0]["n_chars"]) == len("one\ntwo\n")
    assert _CountingHandler.parse_calls == 2
