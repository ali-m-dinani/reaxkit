from __future__ import annotations

from pathlib import Path

import pytest

from reaxkit.engine.reaxff.io.fort7_handler import Fort7Handler
from reaxkit.engine.reaxff.io.xmolout_handler import XmoloutHandler
from reaxkit.engine.reaxff.adapter import ReaxFFAdapter


def _xmolout(n_frames: int) -> str:
    blocks = []
    for frame in range(n_frames):
        blocks.append(
            "\n".join(
                [
                    "2",
                    f"sim {frame * 10} {-10.0 + frame} 8 8 8 90 90 90",
                    f"Al {frame}.0 0 0",
                    f"N {frame + 1}.0 0 0",
                ]
            )
        )
    return "\n".join(blocks) + "\n"


def _fort7(n_frames: int) -> str:
    blocks = []
    for frame in range(n_frames):
        blocks.append(
            "\n".join(
                [
                    f"2 sim Iteration:{frame * 10} #Bonds:1",
                    "1 1 2 1 0.5 0.5 0.0 0.0",
                    "2 2 1 1 0.5 0.5 0.0 0.0",
                    "1.0 0.0 1.0 0.0",
                ]
            )
        )
    return "\n".join(blocks) + "\n"


def test_xmolout_loads_only_requested_frames_in_request_order(tmp_path: Path, monkeypatch):
    path = tmp_path / "xmolout"
    path.write_text(_xmolout(6), encoding="utf-8")
    handler = XmoloutHandler(path, frame_indices=[4, 0, 2])
    monkeypatch.setattr(handler, "_count_lines", lambda: pytest.fail("selective load counted the whole file"))

    assert handler.dataframe()["iter"].tolist() == [40, 0, 20]
    assert handler.metadata()["source_frame_indices"] == [4, 0, 2]
    assert handler.n_frames() == 3
    assert handler.frame(0)["source_index"] == 4
    assert handler.frame(0)["coords"][0, 0] == 4.0


def test_fort7_loads_only_requested_frames_in_request_order(tmp_path: Path, monkeypatch):
    path = tmp_path / "fort.7"
    path.write_text(_fort7(6), encoding="utf-8")
    handler = Fort7Handler(path, frame_indices=[3, 0])
    monkeypatch.setattr(handler, "_count_lines", lambda: pytest.fail("selective load counted the whole file"))

    assert handler.dataframe()["iter"].tolist() == [30, 0]
    assert handler.metadata()["source_frame_indices"] == [3, 0]
    assert handler.n_frames() == 2
    assert handler.frame(0)["atom_num"].tolist() == [1, 2]


def test_full_load_behavior_remains_available(tmp_path: Path):
    xmolout = tmp_path / "xmolout"
    fort7 = tmp_path / "fort.7"
    xmolout.write_text(_xmolout(3), encoding="utf-8")
    fort7.write_text(_fort7(3), encoding="utf-8")

    assert XmoloutHandler(xmolout).dataframe()["iter"].tolist() == [0, 10, 20]
    assert Fort7Handler(fort7).dataframe()["iter"].tolist() == [0, 10, 20]


def test_reaxff_adapter_preserves_source_frame_mapping(tmp_path: Path):
    path = tmp_path / "xmolout"
    path.write_text(_xmolout(5), encoding="utf-8")

    data = ReaxFFAdapter().load_trajectory(
        {"xmolout": str(path), "_frame_indices": [4, 1]},
    )

    assert data.positions.shape == (2, 2, 3)
    assert data.iterations.tolist() == [40, 10]
    assert data.source_frame_indices.tolist() == [4, 1]


def test_reaxff_connectivity_adapter_preserves_source_frame_mapping(tmp_path: Path):
    xmolout = tmp_path / "xmolout"
    fort7 = tmp_path / "fort.7"
    xmolout.write_text(_xmolout(5), encoding="utf-8")
    fort7.write_text(_fort7(5), encoding="utf-8")

    data = ReaxFFAdapter().load_connectivity(
        {
            "fort7": str(fort7),
            "xmolout": str(xmolout),
            "_frame_indices": [3, 0],
        },
    )

    assert data.iterations.tolist() == [30, 0]
    assert data.source_frame_indices.tolist() == [3, 0]
    assert len(data.bond_orders) == 2
