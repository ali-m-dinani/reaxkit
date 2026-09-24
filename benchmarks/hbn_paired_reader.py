"""Drain the production paired reader with a checksum and per-source timings.

Use identical frames with an empty explicit cache directory, with --no-input-cache,
and with the populated directory. OS page-cache state is not controlled here.
This does not run the scientific kernel or copy/modify simulation input files.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from reaxkit.domain.data_models import ElectrostaticsData
from reaxkit.engine.reaxff.adapter import ReaxFFAdapter


def run(input_dir, frames, cache_dir, *, input_cache=True):
    start, stop, step = map(int, frames.split(":"))
    if start < 0 or step <= 0 or stop <= start:
        raise ValueError("Use a finite increasing start:stop:step frame range.")
    requested = list(range(start, stop, step))
    metrics = []
    old_cache_dir = os.environ.get("REAXKIT_FRAME_CACHE_DIR")
    os.environ["REAXKIT_FRAME_CACHE_DIR"] = str(Path(cache_dir).resolve())
    digest = sha256()
    count = 0
    started = perf_counter()
    try:
        args = {"xmolout": str(Path(input_dir).resolve() / "xmolout"),
                "fort7": str(Path(input_dir).resolve() / "fort.7"),
                "_frame_indices": requested, "scope": "total",
                "command": "get-hbn-reference-polarization", "input_cache": input_cache,
                "_reader_timing_callback": lambda **record: metrics.append(record)}
        with closing(ReaxFFAdapter().stream(ElectrostaticsData, args)) as stream:
            for frame in stream:
                source_index = int(frame.trajectory.source_frame_indices[0])
                if count >= len(requested) or source_index != requested[count]:
                    raise ValueError(f"Unexpected output frame {source_index}")
                for value in (frame.trajectory.source_frame_indices, frame.trajectory.iterations,
                              frame.trajectory.positions, frame.trajectory.simulation.cell_lengths,
                              frame.trajectory.simulation.cell_angles, frame.charges.charges):
                    array = np.ascontiguousarray(value)
                    digest.update(str((array.dtype.str, array.shape)).encode("ascii"))
                    digest.update(array.tobytes())
                count += 1
        if count != len(requested):
            raise ValueError(f"Expected {len(requested)} frames; received {count}")
    finally:
        if old_cache_dir is None:
            os.environ.pop("REAXKIT_FRAME_CACHE_DIR", None)
        else:
            os.environ["REAXKIT_FRAME_CACHE_DIR"] = old_cache_dir
    return {"frames": frames, "completed": count, "wall_seconds": perf_counter() - started,
            "checksum": digest.hexdigest(), "input_cache": input_cache,
            "cache_dir": str(Path(cache_dir).resolve()), "sources": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frames", default="0:6144:3")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--no-input-cache", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input, args.frames, args.cache_dir, input_cache=not args.no_input_cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
