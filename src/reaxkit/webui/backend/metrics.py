"""Bounded timing records for GUI profiling; never records scientific payloads."""
from collections import deque
from contextlib import contextmanager
from threading import RLock
from time import perf_counter


class Metrics:
    def __init__(self, capacity=256):
        self.records = deque(maxlen=capacity)
        self.lock = RLock()

    def record(self, stage, duration_ms, **measurements):
        with self.lock:
            self.records.append({'stage': stage, 'duration_ms': duration_ms, **measurements})

    @contextmanager
    def stage(self, name):
        start = perf_counter()
        try:
            yield
        finally:
            self.record(name, (perf_counter() - start) * 1000)

    def snapshot(self):
        with self.lock:
            return list(self.records)
