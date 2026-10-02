"""Shared CLI style selection and workflow propagation."""

import argparse
import importlib
import sys

import pytest

from reaxkit.cli.main import build_parser
from reaxkit.core.runtime.cli_policy import add_execution_arguments
from reaxkit.presentation.plot_styles import resolve_plot_style
from reaxkit.workflows.force_field_opt import get_ffield_opt_plots


@pytest.mark.parametrize("command", ["get_ffield_opt_plots", "get_ffield_opt_eos", "gen-plot"])
def test_plotting_commands_accept_shared_style(command):
    parser = build_parser(command)
    inputs = ["--type", "single", "--file", "data.csv"] if command == "gen-plot" else []
    args = parser.parse_args([command, *inputs, "--plot-style", "publication-bold"])
    assert args.plot_style == "publication-bold"
    with pytest.raises(SystemExit):
        parser.parse_args([command, *inputs, "--plot-style", "unknown"])


def test_style_is_inherited_by_nested_task_and_can_be_overridden():
    parser = argparse.ArgumentParser()
    parser.add_subparsers(dest="task").add_parser("draw")
    add_execution_arguments(parser)
    assert parser.parse_args(["--plot-style", "publication", "draw"]).plot_style == "publication"
    assert parser.parse_args(["--plot-style", "publication", "draw", "--plot-style", "default"]).plot_style == "default"


def test_aggregate_programmatic_invocation_scopes_style(monkeypatch):
    observed = []
    monkeypatch.setattr(get_ffield_opt_plots, "_run_main", lambda command, args: observed.append(resolve_plot_style()))
    get_ffield_opt_plots.run_main("get_ffield_opt_plots", argparse.Namespace(plot_style="publication-bold"))
    assert observed == ["publication-bold"]
    assert resolve_plot_style() == "default"


def test_cli_scopes_style_for_workflow_renderer(monkeypatch, tmp_path):
    from reaxkit.presentation.plot import plot
    import matplotlib.pyplot as plt

    cli = importlib.import_module("reaxkit.cli.main")
    observed = []

    def run(args):
        figure = plot({
            "plot_type": "single_plot", "x": [0, 1], "y": [1, 2],
            "save": tmp_path / "styled.png",
        })
        observed.append(figure.axes[0].lines[0].get_linewidth())
        plt.close(figure)
        return 0

    monkeypatch.setattr(cli, "_direct_command_runner", lambda module, command: run)
    monkeypatch.setattr(sys, "argv", [
        "reaxkit", "get-ffield-opt-plots", "--project-root", str(tmp_path),
        "--plot-style", "publication-bold",
    ])
    assert cli.main(announce=False) == 0
    assert observed == [2.6]
    assert (tmp_path / "styled.png").is_file()
    assert resolve_plot_style() == "default"
