"""Cache lifetimes and active (not generator-suspended) reader measurements."""
from contextlib import ExitStack
from functools import wraps
import io
from time import perf_counter
from reaxkit.core.storage.frame_store import FrameOffset, IndexCoverage


class _MeasuredFile(io.FileIO):
    """Measure raw read calls, including buffered lookahead, not disk traffic."""
    def __init__(self, path, stats):
        super().__init__(path, "rb")
        self.stats = stats

    def readinto(self, buffer):
        started = perf_counter()
        try:
            count = super().readinto(buffer)
        finally:
            self.stats["source_read_seconds"] += perf_counter() - started
            self.stats["source_read_calls"] += 1
        self.stats["source_read_bytes"] += count or 0
        return count


def open_binary_source(handler):
    stats = getattr(handler, "_source_read_stats", None)
    if stats is None:
        stats = handler._source_read_stats = dict(source_read_seconds=0.0,
                                                 source_read_calls=0, source_read_bytes=0,
                                                 source_opens=0)
    stats["source_opens"] += 1
    return io.BufferedReader(_MeasuredFile(handler.path, stats), buffer_size=1024 * 1024)


class FrameWriteBatch:
    """Bound retained payloads by both frame count and estimated array bytes."""
    def __init__(self, store, max_frames=16, max_bytes=8 * 1024 ** 2):
        self.store = store
        self.max_frames = max_frames
        self.max_bytes = max_bytes
        self.pending = {}
        self.bytes = 0

    @staticmethod
    def _size(value):
        if isinstance(value, dict):
            return sum(FrameWriteBatch._size(item) for item in value.values())
        if hasattr(value, "nbytes"):
            return int(value.nbytes)
        if hasattr(value, "memory_usage"):
            return int(value.memory_usage(index=True, deep=True).sum())
        return len(value) if isinstance(value, (str, bytes)) else 64

    def add(self, index, record):
        size = self._size(record)
        if self.pending and self.bytes + size > self.max_bytes:
            self.flush()
        self.pending[index] = record
        self.bytes += size
        if len(self.pending) >= self.max_frames or self.bytes >= self.max_bytes:
            self.flush()

    def flush(self):
        if self.pending:
            self.store.put_frames(self.pending)
            self.pending.clear()
            self.bytes = 0


class OffsetRecorder:
    """Persist only complete frame boundaries discovered by a forward scan."""
    def __init__(self, store):
        self.store = store
        self.pending = []
        self.next_frame = 0

    def __enter__(self):
        return self

    def add(self, frame, start, end, iteration, atoms):
        if self.store is None:
            return
        # A partial/malformed frame must not turn a later offset into a claim
        # of complete coverage over the missing boundary.
        if frame != self.next_frame:
            self.flush()
            self.store = None
            return
        self.pending.append(FrameOffset(frame, start, end, iteration, atoms))
        self.next_frame = frame + 1
        if len(self.pending) >= 128:
            self.flush()

    def flush(self):
        if self.pending:
            last = self.pending[-1]
            if len(self.store.put_offsets(self.pending)) == len(self.pending):
                self.store.set_coverage(IndexCoverage(last.frame_index + 1, last.byte_end, False))
            else:
                self.store = None
            self.pending.clear()

    def __exit__(self, *_):
        self.flush()


def stream_cache_session(method):
    @wraps(method)
    def stream(self, *args, **kwargs):
        stores = []
        elapsed = 0.0
        self._frame_cache_stats = {}
        self._source_read_stats = dict(source_read_seconds=0.0, source_read_calls=0,
                                      source_read_bytes=0, source_opens=0)
        with ExitStack() as stack:
            self._stream_cache_stack = stack
            self._stream_cache_stores = stores
            iterator = method(self, *args, **kwargs)
            try:
                while True:
                    started = perf_counter()
                    try:
                        record = next(iterator)
                    except StopIteration:
                        elapsed += perf_counter() - started
                        break
                    except BaseException:
                        elapsed += perf_counter() - started
                        raise
                    elapsed += perf_counter() - started
                    yield record
            finally:
                started = perf_counter()
                try:
                    iterator.close()
                finally:
                    stack.close()
                    elapsed += perf_counter() - started
                    self._stream_cache_stack = None
                    stats = self._frame_cache_stats
                    stats["reader_active_seconds"] = elapsed
                    try:
                        stats["source_size_bytes"] = self.path.stat().st_size
                    except OSError:
                        stats["source_size_bytes"] = None
                    for name, value in self._source_read_stats.items():
                        stats[name] = stats.get(name, 0) + value
                    stats["source_read_instrumented"] = stats.get("source_opens", 0) > 0
                    stats.setdefault("reader_branch", "indexed" if stores else "generic_text")
                    for store in stores:
                        for name, value in store.stats.items():
                            stats[name] = stats.get(name, 0) + value
                    callback = getattr(self, "_stream_timing_callback", None)
                    if callable(callback):
                        callback(handler=type(self).__name__, source_path=str(self.path), stats=dict(stats))
    return stream


def register_stream_store(handler, store):
    stack = getattr(handler, "_stream_cache_stack", None)
    if stack is not None:
        stack.enter_context(store.session())
        handler._stream_cache_stores.append(store)
    return store
