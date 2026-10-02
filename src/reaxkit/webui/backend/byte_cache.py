"""Thread-safe LRU cache with one shared byte budget for GUI representations."""
from collections import OrderedDict
from collections.abc import MutableMapping
import sys
from threading import RLock


def size_of(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(size_of(k, seen) + size_of(v, seen) for k, v in value.items())
    elif isinstance(value, (tuple, list)):
        size += sum(size_of(v, seen) for v in value)
    return size


class CacheBudget:
    def __init__(self, max_bytes=64 * 1024 * 1024):
        self.max_bytes = max_bytes
        self.bytes = 0
        self.data = OrderedDict()
        self.lock = RLock()

    def namespace(self, name):
        return ByteCache(self, name)


class ByteCache(MutableMapping):
    def __init__(self, budget, name):
        self.budget, self.name = budget, name

    def __getitem__(self, key):
        with self.budget.lock:
            entry = self.budget.data[(self.name, key)]
            self.budget.data.move_to_end((self.name, key))
            return entry[0]

    def __setitem__(self, key, value):
        amount = size_of(value)
        with self.budget.lock:
            self.pop(key, None)
            if amount > self.budget.max_bytes:
                return
            while self.budget.bytes + amount > self.budget.max_bytes:
                _, (_, old_size) = self.budget.data.popitem(last=False)
                self.budget.bytes -= old_size
            self.budget.data[(self.name, key)] = value, amount
            self.budget.bytes += amount

    def __delitem__(self, key):
        with self.budget.lock:
            _, amount = self.budget.data.pop((self.name, key))
            self.budget.bytes -= amount

    def __iter__(self):
        with self.budget.lock:
            return iter([k for ns, k in self.budget.data if ns == self.name])

    def __len__(self):
        with self.budget.lock:
            return sum(ns == self.name for ns, _ in self.budget.data)
