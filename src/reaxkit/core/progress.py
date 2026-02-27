"""Shared progress reporting helpers."""

from __future__ import annotations

from typing import Any, Callable

from reaxkit.core.log import get_logger

ProgressReporter = Callable[[str, int, int, str | None], None]


def noop_reporter(stage: str, current: int, total: int, message: str | None = None) -> None:
    _ = (stage, current, total, message)


def logging_reporter_factory(logger_name: str = __name__) -> ProgressReporter:
    logger = get_logger(logger_name)

    def _report(stage: str, current: int, total: int, message: str | None = None) -> None:
        msg = message or ""
        logger.info("progress stage=%s %d/%d %s", stage, int(current), int(total), msg)

    return _report


def resolve_reporter(args: dict[str, Any]) -> ProgressReporter:
    rep = args.get("reporter")
    if callable(rep):
        return rep
    if args.get("progress"):
        return logging_reporter_factory("reaxkit.progress")
    return noop_reporter
