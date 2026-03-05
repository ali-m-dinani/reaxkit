from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reaxkit.engine.reaxff.io.base import BaseHandler
from reaxkit.engine.reaxff.io.fort7_handler import Fort7Handler
from reaxkit.engine.reaxff.io.fort76_handler import Fort76Handler
from reaxkit.engine.reaxff.io.molfra_handler import MolFraHandler
from reaxkit.engine.reaxff.io.vels_handler import VelsHandler

from .handler_artifacts import EXAMPLES_DIR


def _parquet_available(tmp_path: Path) -> bool:
    probe = tmp_path / "probe.parquet"
    try:
        pd.DataFrame({"x": [1]}).to_parquet(probe, index=False)
        pd.read_parquet(probe)
        return True
    except Exception:
        return False


@pytest.mark.parametrize(
    ("handler_cls", "sample_name", "expected_custom_files"),
    [
        (Fort7Handler, "fort.7", {"fort7_summary.parquet", "fort7_frames.parquet", "fort7.meta.json"}),
        (Fort76Handler, "fort.76", {"fort76_summary.parquet", "fort76.meta.json"}),
        (MolFraHandler, "molfra.out", {"molfra_molecules.parquet", "molfra_totals.parquet", "molfra.meta.json"}),
        (VelsHandler, "vels", {"vels_summary.parquet", "vels.meta.json"}),
    ],
)
def test_heavy_handlers_disk_cache_reuse(
    tmp_path: Path,
    monkeypatch,
    handler_cls,
    sample_name: str,
    expected_custom_files: set[str],
):
    sample_path = EXAMPLES_DIR / sample_name
    if not sample_path.exists():
        pytest.skip(f"Sample input missing: {sample_path}")

    parquet_ok = _parquet_available(tmp_path)
    cache_root = tmp_path / f"cache_{handler_cls.__name__.lower()}"
    monkeypatch.setenv(BaseHandler._CACHE_ENV_VAR, str(cache_root))
    BaseHandler.clear_runtime_cache()

    original_parse = handler_cls._parse
    parse_calls = {"n": 0}

    def _counted_parse(self):
        parse_calls["n"] += 1
        return original_parse(self)

    monkeypatch.setattr(handler_cls, "_parse", _counted_parse)

    h1 = handler_cls(sample_path)
    df1 = h1.dataframe()
    _ = h1.metadata()
    assert parse_calls["n"] == 1

    BaseHandler.clear_runtime_cache()

    h2 = handler_cls(sample_path)
    df2 = h2.dataframe()
    _ = h2.metadata()
    assert parse_calls["n"] == 1
    assert len(df1) == len(df2)
    assert list(df1.columns) == list(df2.columns)

    if isinstance(h1, Fort7Handler):
        assert h1.n_frames() == h2.n_frames()
    if isinstance(h1, Fort76Handler):
        assert h1.n_restraints() == h2.n_restraints()
    if isinstance(h1, MolFraHandler):
        assert len(h1.totals()) == len(h2.totals())
    if isinstance(h1, VelsHandler):
        assert set(h1.sections.keys()) == set(h2.sections.keys())

    entries = list(cache_root.iterdir())
    assert entries
    dirs = [p for p in entries if p.is_dir()]
    assert dirs
    cache_dir = dirs[0]

    if parquet_ok:
        present = {p.name for p in cache_dir.iterdir()}
        if isinstance(h1, VelsHandler):
            assert "sections" in present
            assert "vels_summary.parquet" in present
            assert "vels.meta.json" in present
        else:
            assert expected_custom_files.issubset(present)
    else:
        # Fallback to base structured disk cache when parquet engines are unavailable.
        assert (cache_dir / "dataframe.pkl").exists()
        assert (cache_dir / "meta.json").exists()

