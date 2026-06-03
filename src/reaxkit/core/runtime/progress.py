"""
Shared progress reporting helpers.

**Usage context**

- Import these helpers from ReaxKit core modules when implementing CLI and workflow logic.
- Reuse the public APIs here to keep behavior consistent across commands and engines.
"""

from __future__ import annotations

from typing import Any, Callable

from reaxkit.core.platform.log import get_logger
from tqdm.auto import tqdm

ProgressReporter = Callable[[str, int, int, str | None], None]


def noop_reporter(stage: str, current: int, total: int, message: str | None = None) -> None:
    """
    Noop reporter.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    stage : str
        Input parameter used by this function.
    current : int
        Input parameter used by this function.
    total : int
        Input parameter used by this function.
    message : str | None, optional
        Input parameter used by this function.
    
    Returns
    -----
    None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.runtime.progress import noop_reporter
    # Configure required arguments for your case.
    result = noop_reporter(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    _ = (stage, current, total, message)


def logging_reporter_factory(logger_name: str = __name__) -> ProgressReporter:
    """
    Logging reporter factory.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    logger_name : str, optional
        Input parameter used by this function.
    
    Returns
    -----
    ProgressReporter
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.runtime.progress import logging_reporter_factory
    # Configure required arguments for your case.
    result = logging_reporter_factory(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    logger = get_logger(logger_name)

    def _report(stage: str, current: int, total: int, message: str | None = None) -> None:
        msg = message or ""
        logger.info("progress stage=%s %d/%d %s", stage, int(current), int(total), msg)

    return _report


def tqdm_reporter_factory() -> ProgressReporter:
    """
    Tqdm reporter factory.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    None
    
    Returns
    -----
    ProgressReporter
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.runtime.progress import tqdm_reporter_factory
    # Configure required arguments for your case.
    result = tqdm_reporter_factory(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    bars: dict[str, tqdm] = {}
    last_seen: dict[str, int] = {}
    completed_totals: dict[str, int] = {}

    def _report(stage: str, current: int, total: int, message: str | None = None) -> None:
        key = str(stage or "progress")
        cur = max(0, int(current))
        tot = max(0, int(total))
        msg = (message or "").strip()
        desc = f"{key}: {msg}" if msg else key

        if key not in bars and key in completed_totals and tot > 0 and cur >= tot and completed_totals[key] == tot:
            return

        if key not in bars:
            bars[key] = tqdm(total=tot if tot > 0 else None, desc=desc, unit="step", leave=True, mininterval=0.2)
            last_seen[key] = 0
        bar = bars[key]
        bar.set_description_str(desc)
        if tot > 0 and bar.total is None:
            bar.total = tot
            bar.refresh()

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
    """
    Resolve reporter.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    args : dict[str, Any]
        Input parameter used by this function.
    
    Returns
    -----
    ProgressReporter
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.runtime.progress import resolve_reporter
    # Configure required arguments for your case.
    result = resolve_reporter(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    rep = args.get("reporter")
    if callable(rep):
        return rep
    if args.get("quiet") or args.get("log") == "quiet":
        return noop_reporter
    if args.get("progress"):
        return tqdm_reporter_factory()
    return noop_reporter
