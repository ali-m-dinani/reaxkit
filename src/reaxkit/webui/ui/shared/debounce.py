"""Shared Dash input behavior toggles for Web UI."""

from __future__ import annotations

from dash import dcc


def enable_input_debounce() -> None:
    """Force dcc.Input updates to fire on blur/enter by default."""
    if bool(getattr(dcc, "_reaxkit_input_debounce_enabled", False)):
        return

    base_input = dcc.Input

    def _reaxkit_debounced_input(*args, **kwargs):
        kwargs.setdefault("debounce", True)
        return base_input(*args, **kwargs)

    dcc.Input = _reaxkit_debounced_input
    dcc._reaxkit_input_debounce_enabled = True


__all__ = ["enable_input_debounce"]
