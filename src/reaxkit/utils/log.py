"""
Logging utilities for ReaxKit.

This module provides a small helper for creating consistently formatted
loggers across ReaxKit modules, ensuring uniform log messages in both
CLI workflows and programmatic use.
"""

import logging

def get_logger(name: str) -> logging.Logger:
    """
    Create or retrieve a consistently formatted logger.

    The returned logger uses a standard ReaxKit format and avoids attaching
    duplicate handlers when called multiple times with the same name.

    Parameters
    ----------
    name : str
        Name of the logger, typically ``__name__``.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Parsing started")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers if called multiple times
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
