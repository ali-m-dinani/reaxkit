"""Cross-platform path helpers for filesystem I/O."""

from __future__ import annotations

import os
from pathlib import Path


def io_path(path: str | Path) -> Path:
    """Return an absolute path that supports extended-length I/O on Windows."""

    resolved = Path(path).resolve()
    if os.name != "nt":
        return resolved

    value = str(resolved)
    if value.startswith("\\\\?\\"):
        return resolved
    if value.startswith("\\\\"):
        return Path(f"\\\\?\\UNC\\{value[2:]}")
    return Path(f"\\\\?\\{value}")
