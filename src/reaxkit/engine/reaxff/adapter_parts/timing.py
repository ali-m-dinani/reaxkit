"""Timing and handler-construction helpers for the ReaxFF adapter.

These helpers keep the adapter's loader methods focused on source selection and
data model assembly while preserving centralized timing callback behavior.

**Usage context**

- Loader instrumentation: Emit per-source load timing through callback hooks.
- Handler lifecycle: Build handler instances consistently across loader methods.
- Internal refactoring: Shared by `ReaxFFAdapter` only.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Callable


def _emit_load_timing(
    args: dict,
    *,
    handler: str,
    source_path: Path | str | None,
    seconds: float,
) -> None:
    """Emit a load timing event when a callback is configured in args."""
    cb = args.get("_load_timing_callback")
    if not callable(cb):
        return
    source = None
    source_full = None
    if source_path is not None:
        p = Path(source_path)
        source = p.name
        source_full = str(p)
    cb(handler=str(handler), source=source, source_path=source_full, seconds=float(seconds))


def _build_handler(
    args: dict,
    *,
    handler_name: str,
    source_path: Path | str | None,
    factory: Callable[[], Any],
) -> Any:
    """Build and return a handler instance from the provided factory."""
    _ = (args, handler_name, source_path)
    return factory()


def _time_source(
    args: dict,
    *,
    handler_name: str,
    source_path: Path | str | None,
    loader: Callable[[], Any],
) -> Any:
    """Execute a loader and report elapsed time through the timing callback."""
    t0 = perf_counter()
    out = loader()
    _emit_load_timing(args, handler=handler_name, source_path=source_path, seconds=perf_counter() - t0)
    return out
