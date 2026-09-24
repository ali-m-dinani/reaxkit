"""Bounded synthetic frame-cache scaling benchmark, including cold writes and hits.

This measures cache machinery, not a scientific-command or SLURM speedup.
Run this exact script on both revisions; each case gets a private empty cache.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import platform
import tempfile
from time import perf_counter

import numpy as np

from reaxkit.core.storage.frame_store import FrameStore


def run(count: int, atoms: int, parent: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="hbn-cache-", dir=parent) as directory:
        root = Path(directory)
        source = root / "source"
        source.write_bytes(b"synthetic source identity\n")
        started = perf_counter()
        store = FrameStore.for_source(
            root / "cache", source, engine="reaxff", source_kind="xmolout",
            parser="benchmark", parser_version="1", representation="coordinates",
        )
        payload = {"coordinates": np.arange(atoms * 3, dtype=float).reshape(atoms, 3),
                   "charges": np.zeros(atoms), "source_index": 0}
        session = getattr(store, "session", nullcontext)
        with session():
            for start in range(0, count, 16):
                store.put_frames({i: {**payload, "source_index": i}
                                  for i in range(start, min(start + 16, count))})
        cold = perf_counter() - started
        started = perf_counter()
        loaded = 0
        with session():
            for start in range(0, count, 16):
                records = store.get_frames(range(start, min(start + 16, count)))
                for index, record in records.items():
                    assert record["source_index"] == index
                    np.testing.assert_array_equal(record["coordinates"], payload["coordinates"])
                loaded += len(records)
        assert loaded == count
        return {"frames": count, "atoms": atoms, "cold_seconds": cold,
                "warm_seconds": perf_counter() - started,
                "cache_stats": getattr(store, "stats", {})}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--atoms", type=int, default=5120)
    parser.add_argument("--temporary-root", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {"platform": platform.platform(), "python": platform.python_version(),
               "scope": "synthetic cache only; not end-to-end speedup", "cases": []}
    for count in args.frames:
        case = run(count, args.atoms, args.temporary_root)
        results["cases"].append(case)
        print(json.dumps(case), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
