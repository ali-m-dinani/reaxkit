import numpy as np

from reaxkit.engine.reaxff.io import stream_metrics


def test_active_reader_time_excludes_consumer_work(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(stream_metrics, "perf_counter", lambda: now[0])

    class Reader:
        path = tmp_path / "source"

        @stream_metrics.stream_cache_session
        def records(self):
            for index in range(2):
                now[0] += 1
                yield index

    Reader.path.write_bytes(b"source")
    reader = Reader()
    for _ in reader.records():
        now[0] += 100
    assert reader._frame_cache_stats["reader_active_seconds"] == 2


def test_payload_batch_is_bounded_by_bytes_as_well_as_frames():
    batches = []

    class Store:
        @staticmethod
        def put_frames(frames):
            batches.append(dict(frames))

    batch = stream_metrics.FrameWriteBatch(Store(), max_frames=16, max_bytes=100)
    for index in range(4):
        batch.add(index, {"values": np.zeros(8)})
    batch.flush()
    assert [list(item) for item in batches] == [[0], [1], [2], [3]]
