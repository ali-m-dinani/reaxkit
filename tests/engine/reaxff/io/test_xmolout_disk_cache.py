from __future__ import annotations

from pathlib import Path

import numpy as np

from reaxkit.domain.data_models import ChargeData, ElectrostaticsData, TrajectoryData
from reaxkit.engine.reaxff.adapter import ReaxFFAdapter
from reaxkit.engine.reaxff.io.base import BaseHandler
from reaxkit.engine.reaxff.io.summary_handler import SummaryHandler
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


def test_selected_trajectory_and_summary_cache_are_shared_across_commands(
    tmp_path: Path,
    monkeypatch,
):
    cache_root = tmp_path / "handler_cache"
    monkeypatch.setenv(BaseHandler._CACHE_ENV_VAR, str(cache_root))
    BaseHandler.clear_runtime_cache()

    xmolout_path = tmp_path / "xmolout"
    summary_path = tmp_path / "summary.txt"
    _write_sample_xmolout(xmolout_path)
    summary_path.write_text(
        "\n".join(
            [
                "Iteration Molecules Time Epot Volume Temperature Pressure Density",
                "0 1 0.0 -1.0 1000.0 300.0 1.0 0.5",
                "1 1 0.1 -2.0 1000.0 301.0 1.1 0.5",
                "",
            ]
        ),
        encoding="utf-8",
    )

    parse_calls = {"xmolout": 0, "summary": 0}
    original_xmolout_parse = XmoloutHandler._parse
    original_summary_parse = SummaryHandler._parse

    def _counted_xmolout_parse(self):
        parse_calls["xmolout"] += 1
        return original_xmolout_parse(self)

    def _counted_summary_parse(self):
        parse_calls["summary"] += 1
        return original_summary_parse(self)

    monkeypatch.setattr(XmoloutHandler, "_parse", _counted_xmolout_parse)
    monkeypatch.setattr(SummaryHandler, "_parse", _counted_summary_parse)

    adapter = ReaxFFAdapter()
    common_args = {
        "xmolout": str(xmolout_path),
        "summary": str(summary_path),
        "run_dir": str(tmp_path),
        "_frame_indices": [0],
    }
    monkeypatch.setattr(
        adapter,
        "load_charges",
        lambda args, reporter=None: ChargeData(
            charges=np.zeros((1, 2), dtype=float),
            iterations=np.asarray([0], dtype=int),
        ),
    )
    first = adapter.load(
        ElectrostaticsData,
        {
            **common_args,
            "command": "dynamic-charge-command",
            "_required_data_fields": ("trajectory", "charges"),
        },
    )
    assert first.trajectory.source_frame_indices.tolist() == [0]
    assert parse_calls == {"xmolout": 1, "summary": 1}

    # A new CLI invocation has a fresh process memory cache. The persistent
    # component cache must still be reusable by a different command.
    BaseHandler.clear_runtime_cache()
    second = adapter.load(
        TrajectoryData,
        {**common_args, "command": "formal-charge-command"},
    )

    assert second.source_frame_indices.tolist() == [0]
    assert parse_calls == {"xmolout": 1, "summary": 1}
