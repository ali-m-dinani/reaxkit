"""Measure cold, exact-hit, partial-overlap, and no-overlap frame reads.

Run from the repository root, for example:

    python benchmarks/frame_cache_overlap.py C:/simulation/xmolout --stop 3200

The script reports wall time, parsed-frame count, indexed/source bytes, and
hit/miss counts as JSON. Its cache is isolated under ``--cache-root``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from time import perf_counter

from reaxkit.core.storage.frame_store import clear_frame_cache
from reaxkit.engine.reaxff.io.base import BaseHandler
from reaxkit.engine.reaxff.io.xmolout_handler import XmoloutHandler


def _run(path: Path, cache_root: Path, indices: list[int]) -> dict[str, float | int]:
    BaseHandler.clear_runtime_cache()
    handler = XmoloutHandler(
        path,
        frame_indices=indices,
        frame_cache_root=cache_root,
    )
    started = perf_counter()
    handler.dataframe()
    elapsed = perf_counter() - started
    stats = dict(handler.metadata().get("frame_cache") or {})
    stats["wall_seconds"] = elapsed
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("xmolout", type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path(".frame-cache-benchmark"))
    parser.add_argument("--stop", type=int, default=3200)
    parser.add_argument("--first-step", type=int, default=100)
    parser.add_argument("--second-step", type=int, default=40)
    args = parser.parse_args()

    cache_root = args.cache_root.resolve()
    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True)
    os.environ["REAXKIT_HANDLER_CACHE_DIR"] = str(cache_root / "handlers")
    os.environ["REAXKIT_FRAME_CACHE_DIR"] = str(cache_root)

    first = list(range(0, args.stop, args.first_step))
    second = list(range(0, args.stop, args.second_step))
    no_overlap = list(range(1, args.stop, args.second_step))
    results = {
        "cold": _run(args.xmolout, cache_root, first),
        "exact_hit": _run(args.xmolout, cache_root, list(reversed(first))),
        "partial_overlap": _run(args.xmolout, cache_root, second),
    }
    clear_frame_cache(cache_root)
    results["no_overlap"] = _run(args.xmolout, cache_root, no_overlap)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
