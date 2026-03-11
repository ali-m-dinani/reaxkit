"""Shared progress reporting helpers."""

from __future__ import annotations

from typing import Any, Callable

from reaxkit.core.log import get_logger
from tqdm.auto import tqdm

ProgressReporter = Callable[[str, int, int, str | None], None]


def noop_reporter(stage: str, current: int, total: int, message: str | None = None) -> None:
    _ = (stage, current, total, message)


def logging_reporter_factory(logger_name: str = __name__) -> ProgressReporter:
    logger = get_logger(logger_name)

    def _report(stage: str, current: int, total: int, message: str | None = None) -> None:
        msg = message or ""
        logger.info("progress stage=%s %d/%d %s", stage, int(current), int(total), msg)

    return _report


def tqdm_reporter_factory() -> ProgressReporter:
    bars: dict[str, tqdm] = {}
    last_seen: dict[str, int] = {}
    completed_totals: dict[str, int] = {}

    def _report(stage: str, current: int, total: int, message: str | None = None) -> None:
        key = str(stage or "progress")
        cur = max(0, int(current))
        tot = max(0, int(total))
        msg = (message or "").strip()
        desc = f"{key}: {msg}" if msg else key

        if key in completed_totals and tot > 0 and cur >= tot and completed_totals[key] == tot:
            return

        if key not in bars:
            bars[key] = tqdm(total=tot if tot > 0 else None, desc=desc, unit="step", leave=False, mininterval=0.2)
            last_seen[key] = 0
        bar = bars[key]
        bar.set_description_str(desc)

        prev = int(last_seen.get(key, 0))
        delta = cur - prev
        if delta > 0:
            bar.update(delta)
        last_seen[key] = cur

        if tot > 0 and cur >= tot:
            bar.close()
            bars.pop(key, None)
            last_seen.pop(key, None)
            completed_totals[key] = tot

    return _report


def resolve_reporter(args: dict[str, Any]) -> ProgressReporter:
    rep = args.get("reporter")
    if callable(rep):
        return rep
    if args.get("progress"):
        return tqdm_reporter_factory()
    return noop_reporter
