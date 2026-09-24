"""Reproducible validation benchmark for the bounded frame runtime."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter, process_time
from typing import Iterable

import numpy as np

from reaxkit.core.runtime.execution_contracts import ExecutionPolicy
from reaxkit.core.runtime.frame_pipeline import BoundedFramePipeline


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    frames: int
    payload_bytes: int
    workers: int
    max_in_flight: int
    source_mode: str
    wall_seconds: float
    cpu_seconds: float
    result_digest: str
    peak_in_flight: int
    peak_in_flight_bytes: int
    peak_rss_bytes: int | None = None
    bytes_read: int = 0
    bytes_written: int = 0
    cpu_efficiency: float = 0.0
    frames_per_second: float = 0.0


def _payload(frame: int, payload_bytes: int) -> np.ndarray:
    count = max(1, int(payload_bytes) // np.dtype(np.float64).itemsize)
    return np.full(count, float(frame), dtype=np.float64)


def _memory_source(frames: int, payload_bytes: int) -> Iterable[np.ndarray]:
    for frame in range(int(frames)):
        yield _payload(frame, payload_bytes)


def _file_source(path: Path, frames: int, payload_bytes: int) -> Iterable[np.ndarray]:
    count = max(1, int(payload_bytes) // np.dtype(np.float64).itemsize)
    with path.open("rb") as handle:
        for frame in range(int(frames)):
            yield np.fromfile(handle, dtype=np.float64, count=count)


def _prepare_file(path: Path, frames: int, payload_bytes: int) -> None:
    count = max(1, int(payload_bytes) // np.dtype(np.float64).itemsize)
    expected = int(frames) * count * np.dtype(np.float64).itemsize
    if path.is_file() and path.stat().st_size == expected:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for frame in range(int(frames)):
            _payload(frame, payload_bytes).tofile(handle)


def peak_rss_bytes():
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except ImportError:
        import psutil
        memory = psutil.Process().memory_info()
        return int(getattr(memory, "peak_wset", memory.rss))


def isolated_case(**kwargs):
    """Measure each case in a fresh process so peak RSS is not cumulative."""
    code = ("import json,sys; from dataclasses import asdict; "
            "from reaxkit.core.runtime.benchmark import run_case; "
            "print(json.dumps(asdict(run_case(**json.loads(sys.argv[1])))))")
    completed = subprocess.run([sys.executable, "-c", code, json.dumps(kwargs)],
                               check=True, capture_output=True, text=True)
    return BenchmarkResult(**json.loads(completed.stdout))


def run_case(
    *,
    frames: int,
    payload_bytes: int,
    workers: int,
    max_in_flight: int,
    source_mode: str = "memory",
    workspace: str | Path | None = None,
) -> BenchmarkResult:
    """Run one deterministic workload through the common scheduler."""
    policy = ExecutionPolicy(
        workers=max(1, int(workers)),
        max_in_flight=max(1, int(max_in_flight)),
        backend="thread" if int(workers) > 1 else "serial",
        allocated_cpus=max(1, int(workers)),
        memory_limit_bytes=None,
        estimated_frame_bytes=int(payload_bytes),
        worker_source="benchmark",
        queue_source="benchmark",
        execution_shape="independent_frame_map",
        decision_reason="Phase 8 benchmark",
    )
    if source_mode == "memory":
        source = _memory_source(frames, payload_bytes)
    elif source_mode in {"file_first_read", "file_repeat_read"}:
        root = Path(workspace or ".reaxkit-benchmark")
        path = root / "synthetic_frames.bin"
        _prepare_file(path, frames, payload_bytes)
        source = _file_source(path, frames, payload_bytes)
    else:
        raise ValueError(f"Unsupported benchmark source mode: {source_mode}")

    pipeline = BoundedFramePipeline(policy)
    import psutil
    process = psutil.Process()
    io_before = process.io_counters()

    def kernel(values: np.ndarray) -> tuple[float, float]:
        numeric = np.asarray(values, dtype=np.float64)
        return float(np.sum(numeric)), float(np.dot(numeric, numeric))

    wall_started = perf_counter()
    cpu_started = process_time()
    results = [completed.value for completed in pipeline.map_ordered(source, kernel)]
    cpu_seconds = process_time() - cpu_started
    wall_seconds = perf_counter() - wall_started
    io_after = process.io_counters()
    digest = hashlib.sha256(np.asarray(results, dtype=np.float64).tobytes()).hexdigest()
    return BenchmarkResult(
        frames=int(frames),
        payload_bytes=int(payload_bytes),
        workers=policy.workers,
        max_in_flight=policy.max_in_flight,
        source_mode=source_mode,
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        result_digest=digest,
        peak_in_flight=pipeline.metrics.peak_in_flight,
        peak_in_flight_bytes=pipeline.metrics.peak_in_flight_bytes,
        peak_rss_bytes=peak_rss_bytes(),
        bytes_read=io_after.read_bytes - io_before.read_bytes,
        bytes_written=io_after.write_bytes - io_before.write_bytes,
        cpu_efficiency=cpu_seconds / (wall_seconds * policy.allocated_cpus),
        frames_per_second=int(frames) / wall_seconds,
    )


def run_matrix(
    *,
    frames: int,
    payload_bytes: int,
    worker_counts: Iterable[int],
    queue_multiplier: int = 2,
    workspace: str | Path | None = None,
    include_file_source: bool = True,
    isolate: bool = False,
) -> dict[str, object]:
    """Compare serial/parallel policies and verify deterministic boundedness."""
    modes = ["memory"]
    if include_file_source:
        modes.extend(["file_first_read", "file_repeat_read"])
    cases: list[BenchmarkResult] = []
    for mode in modes:
        for workers in sorted({max(1, int(value)) for value in worker_counts}):
            cases.append(
                (isolated_case if isolate else run_case)(
                    frames=frames,
                    payload_bytes=payload_bytes,
                    workers=workers,
                    max_in_flight=max(1, workers * int(queue_multiplier)),
                    source_mode=mode,
                    workspace=str(workspace) if workspace is not None else None,
                )
            )
    digests = {case.result_digest for case in cases}
    bounded = all(
        case.peak_in_flight <= case.max_in_flight
        and case.peak_in_flight_bytes
        <= case.max_in_flight * case.payload_bytes
        for case in cases
    )
    case_records = [asdict(case) for case in cases]
    recommendations: dict[str, dict[str, float | int]] = {}
    for mode in modes:
        mode_cases = [case for case in cases if case.source_mode == mode]
        baseline = next(case for case in mode_cases if case.workers == min(c.workers for c in mode_cases))
        fastest = min(mode_cases, key=lambda case: case.wall_seconds)
        recommendations[mode] = {
            "fastest_workers": fastest.workers,
            "fastest_wall_seconds": fastest.wall_seconds,
            "speedup_vs_serial": baseline.wall_seconds / fastest.wall_seconds,
        }
        for record in case_records:
            if record["source_mode"] == mode:
                record["speedup_vs_serial"] = baseline.wall_seconds / float(record["wall_seconds"])
    return {
        "schema_version": 1,
        "platform": {
            "python": platform.python_version(),
            "system": platform.platform(),
            "cpu_count": os.cpu_count(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "slurm_mem_per_node_mb": os.environ.get("SLURM_MEM_PER_NODE"),
        },
        "validation": {
            "deterministic": len(digests) == 1,
            "bounded": bounded,
            "case_count": len(cases),
        },
        "recommendations": recommendations,
        "cases": case_records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--payload-mib", type=float, default=1.0)
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--queue-multiplier", type=int, default=2)
    parser.add_argument("--workspace", type=Path, default=Path(".reaxkit-benchmark"))
    parser.add_argument("--output", type=Path, default=Path("runtime_benchmark.json"))
    parser.add_argument("--memory-only", action="store_true")
    parser.add_argument("--in-process", action="store_true", help="Skip subprocess isolation (peak RSS is then cumulative).")
    args = parser.parse_args(argv)
    report = run_matrix(
        frames=max(1, args.frames),
        payload_bytes=max(8, int(args.payload_mib * 1024 * 1024)),
        worker_counts=args.workers,
        queue_multiplier=max(1, args.queue_multiplier),
        workspace=args.workspace,
        include_file_source=not args.memory_only,
        isolate=not args.in_process,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["validation"], sort_keys=True))
    return 0 if all(report["validation"][key] for key in ("deterministic", "bounded")) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["BenchmarkResult", "main", "run_case", "run_matrix"]
