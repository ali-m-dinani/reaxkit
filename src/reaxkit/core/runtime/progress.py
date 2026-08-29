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


class ProgressOperation:
    """Wrap one load/analyze operation and guarantee lifecycle events.

    Fine-grained callbacks from handlers and analyzers are forwarded as-is.
    Operations that do not report their own increments still get an
    indeterminate start event and a deterministic completion event.
    """

    def __init__(
        self,
        reporter: ProgressReporter | None,
        stage: str,
        start_message: str,
        finish_message: str,
    ) -> None:
        self._reporter = reporter if callable(reporter) else noop_reporter
        self._stage = str(stage or "progress")
        self._start_message = str(start_message)
        self._finish_message = str(finish_message)
        self._last_current = 0
        self._last_total = 0

    def __enter__(self) -> "ProgressOperation":
        self._reporter(self._stage, 0, 0, self._start_message)
        return self

    def __call__(self, stage: str, current: int, total: int, message: str | None = None) -> None:
        event_stage = str(stage or self._stage)
        if event_stage == self._stage:
            self._last_current = max(0, int(current))
            self._last_total = max(0, int(total))
        self._reporter(event_stage, int(current), int(total), message)

    def __exit__(self, exc_type, exc, traceback) -> bool:
        _ = traceback
        if self._last_total > 0 and self._last_current >= self._last_total:
            return False
        # Preserve the count accumulated by an indeterminate stream when it
        # becomes determinate at completion. Reporting 1/1 here would make a
        # bar that visibly reached (for example) 500 frames jump backwards.
        total = self._last_total if self._last_total > 0 else max(1, self._last_current)
        message = self._finish_message if exc_type is None else f"Failed: {exc}"
        self._reporter(self._stage, total, total, message)
        return False


def progress_operation(
    reporter: ProgressReporter | None,
    stage: str,
    start_message: str,
    finish_message: str,
) -> ProgressOperation:
    """Create a progress lifecycle wrapper for a blocking operation."""
    return ProgressOperation(reporter, stage, start_message, finish_message)


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
    completed_events: dict[str, tuple[int, int]] = {}

    def _report(stage: str, current: int, total: int, message: str | None = None) -> None:
        key = str(stage or "progress")
        cur = max(0, int(current))
        tot = max(0, int(total))
        msg = (message or "").strip()
        desc = f"{key}: {msg}" if msg else key

        completion = (cur, tot)
        if cur == 0 or (tot > 0 and cur < tot):
            completed_events.pop(key, None)
        if key not in bars and tot > 0 and cur >= tot and completed_events.get(key) == completion:
            return

        if key not in bars:
            bars[key] = tqdm(
                total=tot if tot > 0 else None,
                desc=desc,
                unit="step",
                leave=True,
                mininterval=0.2,
                dynamic_ncols=True,
            )
            last_seen[key] = 0
        bar = bars[key]
        bar.set_description_str(desc)
        if tot > 0 and bar.total != tot:
            bar.total = tot
            bar.refresh()

        prev = int(last_seen.get(key, 0))
        if cur < prev:
            bar.reset(total=tot if tot > 0 else None)
            prev = 0
        delta = cur - prev
        if delta > 0:
            bar.update(delta)
        last_seen[key] = cur

        if tot > 0 and cur >= tot:
            bar.close()
            bars.pop(key, None)
            last_seen.pop(key, None)
            completed_events[key] = completion

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
    if args.get("quiet"):
        return noop_reporter
    if args.get("progress"):
        return tqdm_reporter_factory()
    if args.get("log") == "quiet":
        return noop_reporter
    return noop_reporter
