from __future__ import annotations

from pathlib import Path

from reaxkit.engine.reaxff.io.base import BaseHandler
from reaxkit.engine.reaxff.io.xmolout_handler import XmoloutHandler


def _write_sample_xmolout(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "2",
                "simA 0 -1.000 10.0 10.0 10.0 90.0 90.0 90.0",
                "C 0.0 0.0 0.0",
                "H 1.0 0.0 0.0",
                "2",
                "simA 1 -2.000 10.0 10.0 10.0 90.0 90.0 90.0",
                "C 0.1 0.0 0.0",
                "H 1.1 0.0 0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_xmolout_disk_cache_layout_and_reuse(tmp_path: Path, monkeypatch):
    cache_root = tmp_path / "handler_cache"
    monkeypatch.setenv(BaseHandler._CACHE_ENV_VAR, str(cache_root))
    BaseHandler.clear_runtime_cache()

    xmolout_path = tmp_path / "xmolout"
    _write_sample_xmolout(xmolout_path)

    original_parse = XmoloutHandler._parse
    parse_calls = {"n": 0}

    def _counted_parse(self):
        parse_calls["n"] += 1
        return original_parse(self)

    monkeypatch.setattr(XmoloutHandler, "_parse", _counted_parse)

    h1 = XmoloutHandler(xmolout_path)
    df1 = h1.dataframe()
    assert len(df1) == 2
    assert h1.n_frames() == 2
    assert parse_calls["n"] == 1

    entries = list(cache_root.iterdir())
    assert entries, "Expected at least one on-disk cache entry."
    cache_dirs = [p for p in entries if p.is_dir()]
    has_parquet_layout = any(
        (d / "xmolout_summary.parquet").exists()
        and (d / "xmolout_atoms.parquet").exists()
        and (d / "xmolout.meta.json").exists()
        for d in cache_dirs
    )
    has_base_dir_layout = any(
        (d / "dataframe.pkl").exists() and (d / "meta.json").exists() for d in cache_dirs
    )
    has_pickle_fallback = any(p.suffix == ".pkl" for p in entries)
    assert has_parquet_layout or has_base_dir_layout or has_pickle_fallback

    BaseHandler.clear_runtime_cache()

    h2 = XmoloutHandler(xmolout_path)
    df2 = h2.dataframe()
    assert len(df2) == 2
    assert h2.n_frames() == 2
    assert parse_calls["n"] == 1
