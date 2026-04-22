"""Registry for command-level report payload builders."""

from __future__ import annotations

from pathlib import Path
from typing import Callable


ReportPayloadBuilder = Callable[[str, object, object, Path], dict[str, object] | None]

_REPORT_BUILDERS: dict[str, ReportPayloadBuilder] = {}


def register_report_payload_builder(command: str, builder: ReportPayloadBuilder) -> None:
    """Register a report payload builder for a command."""
    key = str(command).strip()
    if not key:
        raise ValueError("command must be a non-empty string.")
    _REPORT_BUILDERS[key] = builder


def get_report_payload_builder(command: str) -> ReportPayloadBuilder | None:
    """Get a registered report payload builder for a command, if any."""
    key = str(command).strip()
    if not key:
        return None
    return _REPORT_BUILDERS.get(key)


__all__ = [
    "ReportPayloadBuilder",
    "get_report_payload_builder",
    "register_report_payload_builder",
]

