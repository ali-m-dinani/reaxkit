"""Named plot presets shared by workflow commands and the plotting API."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import RLock

from reaxkit.presentation.color_styles import resolve_color_style


@dataclass(frozen=True)
class PlotStyle:
    font_size: float
    label_size: float
    legend_size: float
    weight: str
    linewidth: float
    markersize: float
    axes_width: float
    colors: tuple[str, ...]
    dpi: int = 600


PLOT_STYLES = {
    "default": None,
    "publication": PlotStyle(
        12, 14, 12, "normal", 2.0, 6.0, 1.2,
        ("#0072B2", "#D62728", "#009E73", "#CC79A7", "#E69F00", "#000000"),
    ),
    "publication-bold": PlotStyle(
        14, 17, 13, "bold", 2.6, 7.5, 1.5,
        ("#0057B8", "#D62728", "#008060", "#7B2CBF", "#D87800", "#111111"),
    ),
}
_CURRENT_STYLE = ContextVar("reaxkit_plot_style", default="default")
_RENDER_LOCK = RLock()


def resolve_plot_style(name=None) -> str:
    """Validate a preset name, inheriting the current invocation when omitted."""
    name = _CURRENT_STYLE.get() if name is None else name
    if name not in PLOT_STYLES:
        raise ValueError(f"Unknown plot style {name!r}. Choose from: {', '.join(PLOT_STYLES)}")
    return name


def add_plot_style_argument(parser, *, inherit=False):
    """Add the shared presentation option without duplicating existing flags."""
    if "--plot-style" not in parser._option_string_actions:
        parser.add_argument(
            "--plot-style", choices=tuple(PLOT_STYLES),
            default=argparse.SUPPRESS if inherit else "default",
            help=("Plot appearance: default preserves existing styling; publication uses "
                  "clear fonts and an accessible palette; publication-bold uses larger bold "
                  "text and heavier lines. Example: --plot-style publication-bold, which "
                  "improves readability when figures are reduced in a manuscript."),
        )


@contextmanager
def plot_style_context(name=None):
    """Scope a workflow's selected style without leaking it to later calls."""
    token = _CURRENT_STYLE.set(resolve_plot_style(name))
    try:
        yield
    finally:
        _CURRENT_STYLE.reset(token)


def _size_scale(cfg):
    figsize = cfg.get("figsize")
    if figsize is None and cfg.get("plot_type") == "grouped_bar_plot":
        width = max(8.0, len(cfg.get("labels", [])) * 2.2)
    else:
        width = figsize[0] if figsize is not None else 8.0
    if cfg.get("plot_type") == "multi_subplots":
        return 1.0
    return max(1.0, float(width) / 8.0)


def styled_payload(payload):
    """Copy presentation options; preserve data, custom colors, and explicit sizes."""
    cfg = dict(payload)
    cfg["color_style"] = resolve_color_style(cfg.get("color_style"))
    name = resolve_plot_style(cfg.get("plot_style"))
    cfg["plot_style"] = name
    preset = PLOT_STYLES[name]
    if preset is None:
        return cfg
    scale = _size_scale(cfg)
    cfg.setdefault("dpi", preset.dpi)
    cfg.setdefault("linewidth", preset.linewidth * scale)
    cfg.setdefault("markersize", preset.markersize * scale)
    cfg.setdefault("alpha", 1.0)
    legacy_colors = {"tab:blue": 0, "#1f77b4": 0, "#c0504d": 1}
    if cfg.get("plot_type") in {"scatter3d_points", "scatter3d"}:
        cfg.setdefault("s", cfg["markersize"] ** 2)
    if cfg.get("plot_type") in {"dual_yaxis_plot", "dual_axis"}:
        cfg.setdefault("color1", preset.colors[0])
        cfg.setdefault("color2", preset.colors[1])

    def style_series(series):
        styled = dict(series)
        color = styled.get("color")
        if isinstance(color, str) and color.lower() in legacy_colors:
            styled["color"] = preset.colors[legacy_colors[color.lower()]]
        for option in ("linewidth", "markersize", "alpha"):
            styled.setdefault(option, cfg[option])
        return styled

    cfg = style_series(cfg)
    if cfg.get("series") is not None:
        cfg["series"] = [style_series(series) for series in cfg["series"]]
    if cfg.get("subplots") is not None:
        cfg["subplots"] = [[style_series(series) for series in panel] for panel in cfg["subplots"]]
    return cfg


@contextmanager
def plot_render_context(cfg):
    """Apply temporary Matplotlib defaults under the shared renderer lock."""
    import matplotlib as mpl
    from cycler import cycler

    preset = PLOT_STYLES[resolve_plot_style(cfg.get("plot_style"))]
    settings = {}
    if preset is not None:
        scale = _size_scale(cfg)
        settings = {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": preset.font_size * scale,
            "font.weight": preset.weight,
            "axes.labelsize": preset.label_size * scale,
            "axes.titlesize": preset.label_size * scale,
            "axes.labelweight": preset.weight,
            "axes.titleweight": preset.weight,
            "axes.linewidth": preset.axes_width * scale,
            "axes.prop_cycle": cycler(color=preset.colors),
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "xtick.labelsize": preset.font_size * scale,
            "ytick.labelsize": preset.font_size * scale,
            "xtick.major.width": preset.axes_width * scale,
            "ytick.major.width": preset.axes_width * scale,
            "xtick.major.size": 5 * scale,
            "ytick.major.size": 5 * scale,
            "legend.fontsize": preset.legend_size * scale,
            "legend.frameon": False,
            "lines.linewidth": cfg.get("linewidth", preset.linewidth * scale),
            "lines.markersize": cfg.get("markersize", preset.markersize * scale),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "savefig.dpi": cfg.get("dpi", preset.dpi),
        }
    with _RENDER_LOCK, mpl.rc_context(settings):
        yield


def finish_plot_style(figure, cfg):
    """Apply consistent text and axes styling before final layout and export."""
    preset = PLOT_STYLES[resolve_plot_style(cfg.get("plot_style"))]
    if preset is None:
        return
    scale = _size_scale(cfg)

    def style_text(text, size):
        text.set_fontsize(size * scale)
        text.set_fontweight(preset.weight)
        text.set_color("black")

    for axis in figure.axes:
        for axis_name in ("xaxis", "yaxis", "zaxis"):
            coordinate_axis = getattr(axis, axis_name, None)
            if coordinate_axis is None:
                continue
            style_text(coordinate_axis.label, preset.label_size)
            style_text(coordinate_axis.get_offset_text(), preset.font_size)
            for tick in coordinate_axis.get_ticklabels():
                style_text(tick, preset.font_size)
        style_text(axis.title, preset.label_size)
        for spine in axis.spines.values():
            spine.set_linewidth(preset.axes_width * scale)
            spine.set_edgecolor("black")
        axis.tick_params(width=preset.axes_width * scale, length=5 * scale)
        axis.grid(False)
        legend = axis.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                style_text(text, preset.legend_size)
            style_text(legend.get_title(), preset.legend_size)
        for text in axis.texts:
            style_text(text, preset.font_size)
    for text in figure.texts:
        style_text(text, preset.label_size)
    figure.tight_layout()
