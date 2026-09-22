from __future__ import annotations

from pathlib import Path

import pytest

from reaxkit.domain.data_models import ChargeData, ElectrostaticsData
from reaxkit.engine.reaxff.io.fort7_handler import Fort7Handler
from reaxkit.engine.reaxff.io.xmolout_handler import XmoloutHandler
from reaxkit.engine.reaxff.adapter import ReaxFFAdapter
from reaxkit.engine.reaxff.io.base import BaseHandler


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


def test_numeric_streaming_paths_avoid_per_frame_dataframes(tmp_path: Path):
    xmolout = tmp_path / "xmolout"
    fort7 = tmp_path / "fort.7"
    xmolout.write_text(_xmolout(1), encoding="utf-8")
    fort7.write_text(_fort7(1), encoding="utf-8")

    coordinate_record = next(
        XmoloutHandler(xmolout).stream_file_frames(coordinates_only=True)
    )
    charge_record = next(
        Fort7Handler(fort7).stream_file_frames(charge_arrays_only=True)
    )

    assert "frame" not in coordinate_record
    assert "frame" not in charge_record
    assert coordinate_record["coordinates"].tolist() == [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    assert charge_record["charge_atom_ids"].tolist() == [1, 2]
    assert charge_record["charges"].tolist() == [0.0, 0.0]


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


def test_reaxff_charge_adapter_preserves_source_frame_mapping(tmp_path: Path):
    xmolout = tmp_path / "xmolout"
    fort7 = tmp_path / "fort.7"
    xmolout.write_text(_xmolout(5), encoding="utf-8")
    fort7.write_text(_fort7(5), encoding="utf-8")

    data = ReaxFFAdapter().load(
        ChargeData,
        {
            "fort7": str(fort7),
            "xmolout": str(xmolout),
            "frames": [3, 0],
        },
    )

    assert data.iterations.tolist() == [30, 0]
    assert data.charges.shape == (2, 2)
    assert data.metadata["source_frame_indices"] == [3, 0]


def test_xmolout_reuses_partial_overlap_by_source_frame(tmp_path: Path):
    path = tmp_path / "xmolout"
    cache_root = tmp_path / "cache"
    path.write_text(_xmolout(8), encoding="utf-8")

    first = XmoloutHandler(
        path,
        frame_indices=[0, 2, 4, 6],
        frame_cache_root=cache_root,
    )
    first.dataframe()
    assert first.metadata()["frame_cache"]["parsed_frames"] == 4

    BaseHandler.clear_runtime_cache()
    second = XmoloutHandler(
        path,
        frame_indices=[0, 1, 2, 3, 4, 5, 6, 7],
        frame_cache_root=cache_root,
    )
    assert second.dataframe()["iter"].tolist() == list(range(0, 80, 10))
    stats = second.metadata()["frame_cache"]
    assert stats["hits"] == 4
    assert stats["misses"] == 4
    assert stats["parsed_frames"] == 4


def test_xmolout_indexed_loading_reports_real_frame_progress(tmp_path: Path):
    path = tmp_path / "xmolout"
    cache_root = tmp_path / "cache"
    path.write_text(_xmolout(4), encoding="utf-8")
    events: list[tuple[int, int, str]] = []

    handler = XmoloutHandler(
        path,
        frame_indices=[0, 2, 3],
        frame_cache_root=cache_root,
        reporter=lambda _stage, current, total, message: events.append(
            (current, total, str(message or ""))
        ),
    )
    handler.dataframe()

    frame_events = [event for event in events if event[1] == 7]
    assert frame_events[0][0] == 0
    assert frame_events[-1][0] == 7
    assert [current for current, _, _ in frame_events] == sorted(
        current for current, _, _ in frame_events
    )
    assert all("cache" not in message.lower() for _, _, message in frame_events)


def test_fort7_charge_stream_reports_indexing_and_selected_frame_progress(
        tmp_path: Path,
) -> None:
    path = tmp_path / "fort.7"
    cache_root = tmp_path / "cache"
    path.write_text(_fort7(4), encoding="utf-8")
    events: list[tuple[int, int, str]] = []

    list(
        Fort7Handler(
            path,
            frame_indices=[0, 2, 3],
            frame_cache_root=cache_root,
            reporter=lambda _stage, current, total, message: events.append(
                (current, total, str(message or ""))
            ),
        ).stream_file_frames(charge_arrays_only=True)
    )

    frame_events = [event for event in events if event[1] == 7]
    assert frame_events[0][0] == 0
    assert frame_events[-1][0] == 7
    assert [current for current, _, _ in frame_events] == sorted(
        current for current, _, _ in frame_events
    )
    assert all("cache" not in message.lower() for _, _, message in frame_events)


@pytest.mark.parametrize(
    "command",
    [
        "get-potential-and-electric-field",
        "write-trajectory-with-potential-and-electric-field",
    ],
)
def test_potential_field_streams_report_xmolout_and_fort7_separately(
        tmp_path: Path,
        monkeypatch,
        command: str,
) -> None:
    xmolout = tmp_path / "xmolout"
    fort7 = tmp_path / "fort.7"
    xmolout.write_text(_xmolout(3), encoding="utf-8")
    fort7.write_text(_fort7(3), encoding="utf-8")
    monkeypatch.setenv("REAXKIT_FRAME_CACHE_DIR", str(tmp_path / "cache"))
    events: list[tuple[str, int, int, str]] = []

    frames = list(
        ReaxFFAdapter().iter_data(
            ElectrostaticsData,
            {
                "command": command,
                "xmolout": str(xmolout),
                "fort7": str(fort7),
                "_frame_indices": [0, 2],
                "scope": "total",
            },
            reporter=lambda stage, current, total, message=None: events.append(
                (stage, current, total, str(message or ""))
            ),
        )
    )

    assert len(frames) == 2
    assert ("stream", 1, 1, "Preparing electrostatics input streams") in events
    for stage in ("load xmolout", "load fort.7"):
        source_events = [event for event in events if event[0] == stage]
        assert source_events[0][1:] == (
            0,
            5,
            f"Reading {stage.removeprefix('load ')} frames",
        )
        assert source_events[-1][1] == source_events[-1][2] == 5
        assert [current for _, current, _, _ in source_events] == sorted(
            current for _, current, _, _ in source_events
        )


def test_xmolout_covered_misses_use_offsets_without_rescanning(tmp_path: Path):
    path = tmp_path / "xmolout"
    cache_root = tmp_path / "cache"
    path.write_text(_xmolout(7), encoding="utf-8")

    XmoloutHandler(
        path,
        frame_indices=[6],
        frame_cache_root=cache_root,
    ).dataframe()
    BaseHandler.clear_runtime_cache()
    handler = XmoloutHandler(
        path,
        frame_indices=[1, 3, 5],
        frame_cache_root=cache_root,
    )
    assert handler.dataframe()["iter"].tolist() == [10, 30, 50]
    stats = handler.metadata()["frame_cache"]
    assert stats["indexed_frames"] == 0
    assert stats["index_bytes"] == 0
    assert stats["parsed_frames"] == 3


def test_xmolout_finite_stream_reuses_full_cached_frames(tmp_path: Path):
    path = tmp_path / "xmolout"
    cache_root = tmp_path / "cache"
    path.write_text(_xmolout(5), encoding="utf-8")
    list(
        XmoloutHandler(
            path,
            frame_indices=[0, 2, 4],
            frame_cache_root=cache_root,
        ).stream_file_frames(coordinates_only=True)
    )

    handler = XmoloutHandler(
        path,
        frame_indices=[0, 1, 2, 3, 4],
        frame_cache_root=cache_root,
    )
    records = list(handler.stream_file_frames(coordinates_only=True))
    assert [record["source_index"] for record in records] == [0, 1, 2, 3, 4]
    assert handler._frame_cache_stats["hits"] == 3
    assert handler._frame_cache_stats["parsed_frames"] == 2


def test_input_cache_can_be_disabled_independently(tmp_path: Path):
    path = tmp_path / "xmolout"
    cache_root = tmp_path / "cache"
    path.write_text(_xmolout(3), encoding="utf-8")
    handler = XmoloutHandler(
        path,
        frame_indices=[0, 2],
        frame_cache_root=cache_root,
        input_cache=False,
    )

    assert handler.dataframe()["iter"].tolist() == [0, 20]
    assert "frame_cache" not in handler.metadata()
    assert not (cache_root / "frames").exists()


def test_fort7_charge_stream_reuses_partial_overlap(tmp_path: Path):
    path = tmp_path / "fort.7"
    cache_root = tmp_path / "cache"
    path.write_text(_fort7(6), encoding="utf-8")
    first = Fort7Handler(
        path,
        frame_indices=[0, 2, 4],
        frame_cache_root=cache_root,
    )
    list(first.stream_file_frames(charge_arrays_only=True))

    second = Fort7Handler(
        path,
        frame_indices=[0, 1, 2, 3, 4, 5],
        frame_cache_root=cache_root,
    )
    records = list(second.stream_file_frames(charge_arrays_only=True))
    assert [record["source_index"] for record in records] == list(range(6))
    assert all(record["charges"].tolist() == [0.0, 0.0] for record in records)
    assert second._frame_cache_stats["hits"] == 3
    assert second._frame_cache_stats["parsed_frames"] == 3

    rich_cache = tmp_path / "rich-cache"
    Fort7Handler(
        path,
        frame_indices=[1, 3],
        frame_cache_root=rich_cache,
    ).dataframe()
    narrow = Fort7Handler(
        path,
        frame_indices=[1, 3],
        frame_cache_root=rich_cache,
    )
    list(narrow.stream_file_frames(charge_arrays_only=True))
    assert narrow._frame_cache_stats["hits"] == 2
    assert narrow._frame_cache_stats["parsed_frames"] == 0


def test_distinct_reaxff_data_loaders_share_xmolout_frames(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
):
    xmolout = tmp_path / "xmolout"
    fort7 = tmp_path / "fort.7"
    cache_root = tmp_path / "cache"
    xmolout.write_text(_xmolout(6), encoding="utf-8")
    fort7.write_text(_fort7(6), encoding="utf-8")
    monkeypatch.setenv("REAXKIT_FRAME_CACHE_DIR", str(cache_root))
    parsed: list[int] = []
    original = XmoloutHandler._parse_indexed_frame

    def counted(self, handle, offset):
        parsed.append(offset.frame_index)
        return original(self, handle, offset)

    monkeypatch.setattr(XmoloutHandler, "_parse_indexed_frame", counted)
    adapter = ReaxFFAdapter()
    trajectory = adapter.load_trajectory(
        {"xmolout": str(xmolout), "_frame_indices": [0, 2, 4]},
    )
    assert trajectory.source_frame_indices.tolist() == [0, 2, 4]
    assert parsed == [0, 2, 4]

    BaseHandler.clear_runtime_cache()
    parsed.clear()
    combined = adapter.load_connectivity_trajectory(
        {
            "xmolout": str(xmolout),
            "fort7": str(fort7),
            "_frame_indices": [0, 1, 2, 3, 4, 5],
        },
    )
    assert combined.trajectory.source_frame_indices.tolist() == list(range(6))
    assert parsed == [1, 3, 5]


def test_selection_normalization_preserves_order_and_handles_edges(tmp_path: Path):
    path = tmp_path / "xmolout"
    path.write_text(_xmolout(5), encoding="utf-8")

    selected = XmoloutHandler(
        path,
        frame_indices=[4, 0, 4, -1, 2, 99],
        frame_cache_root=tmp_path / "cache",
    )
    assert selected.metadata()["source_frame_indices"] == [4, 0, 2]
    assert selected.dataframe()["iter"].tolist() == [40, 0, 20]

    empty = XmoloutHandler(
        path,
        frame_indices=[],
        frame_cache_root=tmp_path / "empty-cache",
    )
    assert empty.dataframe().empty
    assert empty.metadata()["source_frame_indices"] == []
