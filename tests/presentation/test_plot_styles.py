"""Style behavior at the public rendering boundary."""

from copy import deepcopy

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_rgba

from reaxkit.presentation.plot import plot
from reaxkit.presentation.plot_styles import PLOT_STYLES, plot_style_context, resolve_plot_style


@pytest.fixture
def curve():
    return {
        "plot_type": "single_plot",
        "series": [
            {"x": [1, 2, 3], "y": [2, 0, 3], "label": "ReaxFF", "marker": "o", "color": "tab:blue"},
            {"x": [1, 2, 3], "y": [1, 0, 2], "label": "QM/Literature", "marker": "o", "color": "#C0504D"},
        ],
        "xlabel": "Volume", "ylabel": "Energy", "title": "EOS",
        "legend": True, "figsize": (6, 5),
    }


@pytest.mark.parametrize("name", ["publication", "publication-bold"])
def test_presets_style_artists_without_mutating_data(name, curve, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    original = deepcopy(curve)
    before = dict(matplotlib.rcParams)
    figure = plot(curve, style=name)
    preset = PLOT_STYLES[name]
    axis = figure.axes[0]
    for index, line in enumerate(axis.lines):
        assert list(line.get_ydata()) == curve["series"][index]["y"]
        assert line.get_color() == preset.colors[index]
        assert line.get_linewidth() == preset.linewidth
        assert line.get_markersize() == preset.markersize
    assert axis.xaxis.label.get_fontsize() == preset.label_size
    assert axis.yaxis.label.get_fontweight() == preset.weight
    assert axis.title.get_fontsize() == preset.label_size
    assert all(text.get_fontweight() == preset.weight for text in axis.get_xticklabels())
    assert axis.get_legend().get_texts()[0].get_fontsize() == preset.legend_size
    assert curve == original
    assert dict(matplotlib.rcParams) == before
    plt.close(figure)


def test_default_preserves_existing_appearance(curve, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    figure = plot(curve, style="default")
    line = figure.axes[0].lines[0]
    assert line.get_color() == "tab:blue"
    assert line.get_linewidth() == 1.2
    assert line.get_markersize() == 4
    plt.close(figure)


def test_explicit_series_sizes_and_custom_colors_override_preset(curve, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    curve["series"][0].update(color="#123456", linewidth=4.5, markersize=10)
    figure = plot(curve, style={"plot_style": "publication", "linewidth": 3.5})
    first, second = figure.axes[0].lines
    assert first.get_color() == "#123456"
    assert first.get_linewidth() == 4.5
    assert first.get_markersize() == 10
    assert second.get_linewidth() == 3.5
    plt.close(figure)


def test_grouped_bars_use_opaque_preset_colors_and_keep_values(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    figure = plot({
        "plot_type": "grouped_bar_plot", "plot_style": "publication-bold",
        "labels": ["A", "B"],
        "series": [
            {"values": [-2, 3], "label": "ReaxFF", "color": "tab:blue"},
            {"values": [-1, 4], "label": "QM/Literature", "color": "#C0504D"},
        ],
    })
    axis = figure.axes[0]
    assert [bar.get_height() for bar in axis.patches] == [-2, 3, -1, 4]
    assert axis.patches[0].get_facecolor() == to_rgba(PLOT_STYLES["publication-bold"].colors[0])
    assert axis.patches[2].get_facecolor() == to_rgba(PLOT_STYLES["publication-bold"].colors[1])
    assert all(bar.get_alpha() == 1 for bar in axis.patches)
    assert not any(line.get_visible() for line in axis.get_ygridlines())
    plt.close(figure)


def test_subplot_pages_share_line_and_legend_style(curve, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    figures = plot({
        "plot_type": "multi_subplots", "subplots": [curve["series"]] * 3,
        "grid": "1x2", "plot_style": "publication-bold", "legend": True,
    })
    assert len(figures) == 2
    for figure in figures:
        for axis in figure.axes:
            if not axis.axison:
                continue
            assert axis.lines[0].get_marker() == "o"
            assert axis.lines[0].get_color() == PLOT_STYLES["publication-bold"].colors[0]
            assert axis.lines[0].get_linewidth() == 2.6
            assert axis.get_legend().get_texts()[0].get_fontweight() == "bold"
        plt.close(figure)


def test_style_context_and_rcparams_reset_even_after_failure(curve, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    before = dict(matplotlib.rcParams)
    with pytest.raises(ValueError, match="Provide"):
        with plot_style_context("publication-bold"):
            assert resolve_plot_style() == "publication-bold"
            figure = plot(curve)
            assert figure.axes[0].lines[0].get_linewidth() == 2.6
            plt.close(figure)
            plot({"plot_type": "single_plot"})
    assert resolve_plot_style() == "default"
    assert dict(matplotlib.rcParams) == before
    plt.close("all")


def test_explicit_default_overrides_workflow_style(curve, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    with plot_style_context("publication-bold"):
        figure = plot({**curve, "plot_style": "default"})
    assert figure.axes[0].lines[0].get_color() == "tab:blue"
    plt.close(figure)


def test_export_uses_preset_resolution_and_editable_vector_fonts(curve, monkeypatch, tmp_path):
    saves = []
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda self, path, **kwargs: saves.append(
        (path, kwargs, matplotlib.rcParams["pdf.fonttype"], matplotlib.rcParams["svg.fonttype"],
         self.axes[0].title.get_fontweight())
    ))
    plot({**curve, "save": tmp_path / "eos.png"}, style="publication-bold")
    assert saves[0][1]["dpi"] == 600
    assert saves[0][2:] == (42, "none", "bold")
    plot({**curve, "save": tmp_path / "eos.png", "dpi": 450}, style="publication")
    assert saves[1][1]["dpi"] == 450


def test_invalid_preset_fails_before_rendering(curve):
    with pytest.raises(ValueError, match="Unknown plot style"):
        plot(curve, style="invalid")


def test_svg_retains_editable_labels(curve, tmp_path):
    from xml.etree import ElementTree

    destination = tmp_path / "eos.svg"
    plot({**curve, "save": destination}, style="publication-bold")
    document = ElementTree.parse(destination)
    labels = [text.text for text in document.findall(".//{http://www.w3.org/2000/svg}text")]
    assert "Volume" in labels
    assert "Energy" in labels
    assert "ReaxFF" in labels


def test_errorbar_series_obey_preset_and_custom_overrides(curve, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    curve["plot_type"] = "errorbar_plot"
    curve["series"][0].update(yerr=[0.1, 0.2, 0.1], color="#123456", linewidth=4)
    figure = plot(curve, style="publication")
    first = figure.axes[0].containers[0].lines[0]
    second = figure.axes[0].containers[1].lines[0]
    assert first.get_color() == "#123456"
    assert first.get_linewidth() == 4
    assert second.get_color() == PLOT_STYLES["publication"].colors[1]
    plt.close(figure)
