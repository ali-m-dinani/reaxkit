"""Shared color treatments preserve data and style scope."""

import argparse
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import pytest

from reaxkit.presentation.color_styles import color_style_context, resolve_color_style
from reaxkit.presentation.plot import plot
from reaxkit.cli.main import build_parser
from reaxkit.workflows.force_field_opt import get_ffield_opt_plots


@pytest.mark.parametrize("plot_style", ["default", "publication", "publication-bold"])
@pytest.mark.parametrize("color_style", ["default", "light-fill"])
@pytest.mark.parametrize("kind", ["single_plot", "grouped_bar_plot", "multi_subplots"])
def test_color_treatments(kind, color_style, plot_style, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    series = [{"x": [0, 1], "y": [-2, 3], "values": [-2, 3], "label": "ReaxFF",
               "color": "#0057B8", "marker": "o"}]
    payload = {"plot_type": kind, "series": series, "subplots": [series], "grid": "1x1",
               "labels": ["A", "B"], "legend": True, "plot_style": plot_style,
               "color_style": color_style}
    original = deepcopy(payload)
    result = plot(payload)
    figure = result[0] if isinstance(result, list) else result
    try:
        axis = figure.axes[0]
        artist = axis.patches[0] if kind == "grouped_bar_plot" else axis.lines[0]
        if kind == "grouped_bar_plot":
            face, edge = artist.get_facecolor(), artist.get_edgecolor()
            assert artist.get_height() == -2
        else:
            face, edge = to_rgba(artist.get_markerfacecolor()), to_rgba(artist.get_markeredgecolor())
            assert list(artist.get_ydata()) == [-2, 3]
            assert artist.get_color() == "#0057B8"
        if color_style == "light-fill":
            assert edge[:3] == to_rgba("#0057B8")[:3]
            assert all(fill > border for fill, border in zip(face[:3], edge[:3]))
            handle = axis.get_legend().legend_handles[0]
            if kind == "grouped_bar_plot":
                assert artist.get_linewidth() == 1.7
                assert handle.get_linewidth() == 1.7
                assert handle.get_facecolor() == face
                assert handle.get_edgecolor() == edge
            else:
                assert to_rgba(handle.get_markerfacecolor()) == face
        else:
            assert face[:3] == to_rgba("#0057B8")[:3]
        assert payload == original
    finally:
        plt.close(figure)


def test_color_scope_and_explicit_override():
    with pytest.raises(RuntimeError):
        with color_style_context("light-fill"):
            assert resolve_color_style() == "light-fill"
            assert resolve_color_style("default") == "default"
            raise RuntimeError
    assert resolve_color_style() == "default"
    with pytest.raises(ValueError, match="Unknown color style"):
        plot({"plot_type": "single_plot", "color_style": "unknown"})


@pytest.mark.parametrize("command,inputs", [
    ("get_ffield_opt_plots", ["--project-root", "."]),
    ("plot-from-excel", ["--input", "input.xlsx"]),
])
def test_cli_color_option(command, inputs):
    parser = build_parser(command)
    assert parser.parse_args([command, *inputs]).color_style == "default"
    assert parser.parse_args([command, *inputs, "--color-style", "light-fill"]).color_style == "light-fill"
    with pytest.raises(SystemExit):
        parser.parse_args([command, *inputs, "--color-style", "unknown"])


def test_programmatic_aggregate_scopes_color_style(monkeypatch):
    observed = []
    monkeypatch.setattr(get_ffield_opt_plots, "_run_main",
                        lambda command, args: observed.append(resolve_color_style()))
    get_ffield_opt_plots.run_main("get_ffield_opt_plots", argparse.Namespace(color_style="light-fill"))
    assert observed == ["light-fill"]
    assert resolve_color_style() == "default"
