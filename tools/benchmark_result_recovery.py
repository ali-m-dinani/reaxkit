"""Measure result storage, export and validation separately from input loading."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import psutil

from reaxkit.core.runtime.result_store import ResultStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--buffer-mb", type=float, default=4)
    args = parser.parse_args()
    started = time.perf_counter()
    process = psutil.Process()
    peak = process.memory_info().rss
    with ResultStore(args.directory, {"benchmark": 1}, buffer_bytes=int(args.buffer_mb * 1024 * 1024)) as store:
        for frame in range(args.frames):
            table = pd.DataFrame({"frame_index": np.full(args.rows, frame), "atom_id": np.arange(args.rows),
                                  "value": np.arange(args.rows, dtype=float) / 3})
            store.append(frame, frame, {"table": table})
            peak = max(peak, process.memory_info().rss)
        store.transition("analysis_complete")
        calculation_seconds = time.perf_counter() - started
        started = time.perf_counter()
        store.table("table").to_csv(args.directory / "result.csv")
        export_seconds = time.perf_counter() - started
        peak = max(peak, process.memory_info().rss)
        store.transition("complete")
    started = time.perf_counter()
    with ResultStore(args.directory, {"benchmark": 1}, resume=True) as store:
        batches = store.manifest["batch_count"]
    report = {"frames": args.frames, "rows_per_frame": args.rows, "buffer_mb": args.buffer_mb,
              "sampled_peak_rss_bytes": peak, "batches": batches, "write_seconds": calculation_seconds,
              "export_seconds": export_seconds, "resume_validation_seconds": time.perf_counter() - started,
              "disk_bytes": sum(path.stat().st_size for path in args.directory.rglob("*") if path.is_file())}
    (args.directory / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
