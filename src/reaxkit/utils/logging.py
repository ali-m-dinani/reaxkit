"""Lightweight logging utility providing consistently formatted loggers across ReaxKit modules."""
import logging

def get_logger(name: str) -> logging.Logger:
    """Return a logger with a consistent format."""
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
