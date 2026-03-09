"""Presentation utilities for plotting, media, and user-facing units."""

from reaxkit.presentation.specs import PresentationSpec

__all__ = ["plot", "present_result", "PresentationSpec"]


def __getattr__(name: str):
    if name == "plot":
        from reaxkit.presentation.plot import plot

        return plot
    if name == "present_result":
        from reaxkit.presentation.dispatcher import present_result

        return present_result
    raise AttributeError(name)
