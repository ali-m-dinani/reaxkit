"""
Registry for command-level report payload builders.

**Usage context**

- Import these helpers from presentation workflows that produce tables, files, or plots.
- Reuse the public APIs here to keep output formatting and artifact behavior consistent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable


ReportPayloadBuilder = Callable[[str, object, object, Path], dict[str, object] | None]

_REPORT_BUILDERS: dict[str, ReportPayloadBuilder] = {}


def register_report_payload_builder(command: str, builder: ReportPayloadBuilder) -> None:
    """
    Register a report payload builder for a command.
    
    This function is part of the ReaxKit presentation API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    command : str
        Input parameter used by this function.
    builder : ReportPayloadBuilder
        Input parameter used by this function.
    
    Returns
    -----
    None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.presentation.report_registry import register_report_payload_builder
    result = register_report_payload_builder(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    key = str(command).strip()
    if not key:
        raise ValueError("command must be a non-empty string.")
    _REPORT_BUILDERS[key] = builder


def get_report_payload_builder(command: str) -> ReportPayloadBuilder | None:
    """
    Get a registered report payload builder for a command, if any.
    
    This function is part of the ReaxKit presentation API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    command : str
        Input parameter used by this function.
    
    Returns
    -----
    ReportPayloadBuilder | None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.presentation.report_registry import get_report_payload_builder
    result = get_report_payload_builder(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    key = str(command).strip()
    if not key:
        return None
    return _REPORT_BUILDERS.get(key)


__all__ = [
    "ReportPayloadBuilder",
    "get_report_payload_builder",
    "register_report_payload_builder",
]

