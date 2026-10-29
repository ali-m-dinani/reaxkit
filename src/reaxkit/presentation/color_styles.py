"""Color treatments independent of typography and plot sizing."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar


COLOR_STYLES = ("default", "light-fill")
_CURRENT_STYLE = ContextVar("reaxkit_color_style", default="default")


def resolve_color_style(name=None):
    """Validate a color treatment, inheriting the current workflow when omitted."""
    name = _CURRENT_STYLE.get() if name is None else name
    if name not in COLOR_STYLES:
        raise ValueError(f"Unknown color style {name!r}. Choose from: {', '.join(COLOR_STYLES)}")
    return name


def add_color_style_argument(parser, *, inherit=False):
    """Register the shared color option once per parser."""
    if "--color-style" not in parser._option_string_actions:
        parser.add_argument(
            "--color-style", choices=COLOR_STYLES,
            default=argparse.SUPPRESS if inherit else "default",
            help=("Color treatment: default preserves existing colors; light-fill uses "
                  "lighter bar and marker interiors with the original outlines and lines. "
                  "Example: --color-style light-fill, which adds contrast between fills and borders."),
        )


@contextmanager
def color_style_context(name=None):
    """Scope color selection without affecting subsequent invocations."""
    token = _CURRENT_STYLE.set(resolve_color_style(name))
    try:
        yield
    finally:
        _CURRENT_STYLE.reset(token)


def finish_color_style(figure, cfg):
    """Tint bar and line-marker interiors, including their legend handles."""
    if resolve_color_style(cfg.get("color_style")) == "default":
        return
    from matplotlib.colors import to_rgba
    from matplotlib.container import BarContainer
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    def lighter(color):
        red, green, blue, alpha = to_rgba(color)
        return tuple(channel + (1 - channel) * 0.35 for channel in (red, green, blue)) + (alpha,)

    def style_artist(artist):
        if isinstance(artist, Rectangle):
            color = artist.get_facecolor()
            artist.set_edgecolor(color)
            artist.set_facecolor(lighter(color))
            artist.set_linewidth(1.7)
        elif isinstance(artist, Line2D) and artist.get_marker() not in (None, "None", "", " "):
            color = artist.get_color()
            artist.set_markeredgecolor(color)
            artist.set_markerfacecolor(lighter(color))

    for axis in figure.axes:
        for container in axis.containers:
            if isinstance(container, BarContainer):
                for bar in container.patches:
                    style_artist(bar)
        for line in axis.lines:
            style_artist(line)
        legend = axis.get_legend()
        if legend is not None:
            for handle in legend.legend_handles:
                style_artist(handle)
