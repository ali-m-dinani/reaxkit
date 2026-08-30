from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd

from reaxkit.presentation.dispatcher import present_result
from reaxkit.presentation.plot.renderers.single import SinglePlotRenderer


def test_present_result_prints_table_when_no_output_flags(capsys):
    args = argparse.Namespace(plot=None, show=False, save=None, export=None)
    result = SimpleNamespace(table=pd.DataFrame({"value": [1, 2]}))

    present_result("demo", result, args)

    out = capsys.readouterr().out
    assert "value" in out
    assert "1" in out


def test_present_result_saves_plot_payload_batch_to_directory(monkeypatch, tmp_path):
    rendered = []
    monkeypatch.setattr(
        "reaxkit.presentation.dispatcher.render_plot",
        lambda payload: rendered.append(payload),
    )
    args = argparse.Namespace(
        plot="single",
        show=False,
        save=str(tmp_path / "eos_plots"),
        export=None,
        project_root=str(tmp_path / "workspace"),
        run_id="run-1",
        analysis_id="analysis-1",
    )
    result = SimpleNamespace(table=pd.DataFrame({"value": [1, 2]}))

    present_result(
        "demo",
        result,
        args,
        plot_payload_builder=lambda _command, _result, _args: [
            {
                "plot_type": "single_plot",
                "x": [1, 2],
                "y": [2, 3],
                "filename": "first.png",
                "subdirectory": "material_a",
            },
            {
                "plot_type": "single_plot",
                "x": [1, 2],
                "y": [3, 4],
                "filename": "second.png",
            },
        ],
    )

    assert [Path(payload["save"]).name for payload in rendered] == [
        "first.png",
        "second.png",
    ]
    assert Path(rendered[0]["save"]).parent == tmp_path / "eos_plots" / "material_a"
    assert Path(rendered[1]["save"]).parent == tmp_path / "eos_plots"


def test_present_result_copy_to_dot_copies_saved_plot(monkeypatch, tmp_path):
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)

    def fake_render(payload):
        output = Path(payload["save"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"plot-data")

    monkeypatch.setattr(
        "reaxkit.presentation.dispatcher.render_plot",
        fake_render,
    )
    args = argparse.Namespace(
        plot="single",
        show=False,
        save="electric-field.png",
        export=None,
        copy_to_dot=True,
        project_root=str(tmp_path / "workspace"),
        run_id="run-1",
        analysis_id="analysis-1",
    )
    result = SimpleNamespace(table=pd.DataFrame({"value": [1, 2]}))

    present_result(
        "get_electric_field",
        result,
        args,
        plot_payload_builder=lambda _command, _result, _args: {
            "plot_type": "single_plot",
            "x": [0, 1],
            "y": [1, 2],
        },
    )

    copied = working_dir / "electric-field.png"
    assert copied.read_bytes() == b"plot-data"


def test_single_plot_renderer_applies_per_series_colors(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)

    figure = SinglePlotRenderer().render(
        {
            "series": [
                {"x": [1, 2], "y": [2, 3], "color": "tab:blue"},
                {"x": [1, 2], "y": [3, 4], "color": "#C0504D"},
            ]
        }
    )

    assert [line.get_color() for line in figure.axes[0].lines] == [
        "tab:blue",
        "#C0504D",
    ]
    plt.close(figure)
