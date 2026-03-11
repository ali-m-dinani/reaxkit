"""Logging utilities for ReaxKit."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from threading import Lock


_LOGGER_LOCK = Lock()
_REGISTERED_LOGGERS: set[logging.Logger] = set()
_GLOBAL_FILE_HANDLER: logging.Handler | None = None
_RUN_FILE_HANDLER: logging.Handler | None = None
_CURRENT_SESSION_ID: str | None = None
_CURRENT_LOGS_ROOT: Path | None = None


def _resolve_level(level: str | int | None) -> int | None:
    if level is None:
        return None
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), None)


def _formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def configure_file_logging(project_root: str | Path, *, session_id: str | None = None) -> str:
    """
    Configure default file logging for all ReaxKit loggers.

    Creates:
      - logs/general/reaxkit_general.log
      - logs/general/run_<session_id>.general.log
    """
    global _GLOBAL_FILE_HANDLER, _RUN_FILE_HANDLER, _CURRENT_SESSION_ID, _CURRENT_LOGS_ROOT
    root = Path(project_root)
    logs_root = root / "logs" / "general"
    logs_root.mkdir(parents=True, exist_ok=True)
    sid = session_id or datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    with _LOGGER_LOCK:
        if _GLOBAL_FILE_HANDLER is None or _CURRENT_LOGS_ROOT != logs_root:
            if _GLOBAL_FILE_HANDLER is not None:
                try:
                    _GLOBAL_FILE_HANDLER.close()
                except Exception:
                    pass
            global_path = logs_root / "reaxkit_general.log"
            gh = logging.FileHandler(global_path, encoding="utf-8")
            gh.setFormatter(_formatter())
            _GLOBAL_FILE_HANDLER = gh

        if _RUN_FILE_HANDLER is not None and _CURRENT_SESSION_ID != sid:
            for logger in _REGISTERED_LOGGERS:
                try:
                    logger.removeHandler(_RUN_FILE_HANDLER)
                except Exception:
                    pass
            try:
                _RUN_FILE_HANDLER.close()
            except Exception:
                pass
            _RUN_FILE_HANDLER = None

        if _RUN_FILE_HANDLER is None:
            run_path = logs_root / f"run_{sid}.general.log"
            rh = logging.FileHandler(run_path, encoding="utf-8")
            rh.setFormatter(_formatter())
            _RUN_FILE_HANDLER = rh

        _CURRENT_SESSION_ID = sid
        _CURRENT_LOGS_ROOT = logs_root

        for logger in _REGISTERED_LOGGERS:
            if _GLOBAL_FILE_HANDLER and _GLOBAL_FILE_HANDLER not in logger.handlers:
                logger.addHandler(_GLOBAL_FILE_HANDLER)
            if _RUN_FILE_HANDLER and _RUN_FILE_HANDLER not in logger.handlers:
                logger.addHandler(_RUN_FILE_HANDLER)

    return sid


def get_logger(name: str, *, level: str | int | None = None) -> logging.Logger:
    """Create or retrieve a consistently formatted logger."""
    logger = logging.getLogger(name)
    with _LOGGER_LOCK:
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
            logger.propagate = False
            logger.setLevel(logging.INFO)

        if _GLOBAL_FILE_HANDLER and _GLOBAL_FILE_HANDLER not in logger.handlers:
            logger.addHandler(_GLOBAL_FILE_HANDLER)
        if _RUN_FILE_HANDLER and _RUN_FILE_HANDLER not in logger.handlers:
            logger.addHandler(_RUN_FILE_HANDLER)
        _REGISTERED_LOGGERS.add(logger)

    lvl = _resolve_level(level)
    if lvl is not None:
        logger.setLevel(lvl)
    return logger
