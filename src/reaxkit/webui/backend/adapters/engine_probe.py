"""Engine detection adapter for dataset nodes."""

from __future__ import annotations

from pathlib import Path


def detect_engine(path: str, engine_override: str | None = None) -> str:
    """Resolve engine name from path with optional explicit override."""
    try:
        from reaxkit.core.platform.engine_resolver import resolve_engine
        import reaxkit.engine  # noqa: F401  (register adapters)

        adapter = resolve_engine(str(Path(path)), engine=engine_override)
        return str(getattr(adapter, "name", adapter.__class__.__name__.replace("Adapter", "").lower()))
    except Exception:
        # Fallback keeps UI usable in environments missing heavy scientific deps.
        if engine_override:
            return str(engine_override)
        p = Path(path)
        if (p / "xmolout").exists() or (p / "fort.7").exists():
            return "reaxff"
        return "unknown"
