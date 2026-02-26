"""Logging utilities for ReaxKit."""

from __future__ import annotations

import logging


def _resolve_level(level: str | int | None) -> int | None:
    if level is None:
        return None
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), None)


def get_logger(name: str, *, level: str | int | None = None) -> logging.Logger:
    """Create or retrieve a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        logger.setLevel(logging.INFO)
    lvl = _resolve_level(level)
    if lvl is not None:
        logger.setLevel(lvl)
    return logger
