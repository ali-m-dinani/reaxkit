from __future__ import annotations

import argparse
from types import SimpleNamespace

import pandas as pd

from reaxkit.analysis.stress_strain.z_binned_deformation_gradient_strain import (
    ZBinnedDeformationGradientStrainRequest,
)
from reaxkit.analysis.stress_strain.z_binned_strain_using_top_bottom_atoms import (
    ZBinnedTopBottomStrainRequest,
)
from reaxkit.core.registry.analysis_cli_routing_registry import get_registered_analysis_commands
from reaxkit.workflows.stress_strain import z_binned_strain_workflow as workflow


def test_top_bottom_parser_and_request_builder() -> None:
    parser = workflow.build_parser(
        argparse.ArgumentParser(),
        command="z-binned-strain-using-top-bottom-atoms",
    )
    args = parser.parse_args(
        ["--z-bins", "12", "--atom-types", "Al", "N", "--frames", "0:21:10", "--periodic", "xy", "--n-extreme-atoms", "8", "--gen-plots", "--y-scale", "global", "--plot-every", "2"]
    )
    request = workflow.REQUEST_BUILDERS[args.command](args)

    assert args.command == "get_z_binned_top_bottom_strain"
    assert isinstance(request, ZBinnedTopBottomStrainRequest)
    assert request.selected_frames == [0, 10, 20]
    assert request.n_extreme_atoms == 8
    assert args.gen_plots is True
    assert args.y_scale == "global"
    assert args.plot_every == 2


def test_deformation_gradient_parser_and_routes() -> None:
    parser = workflow.build_parser(
        argparse.ArgumentParser(),
        command="z_binned_deformation_gradient_strain",
    )
    args = parser.parse_args(["--z-bins", "6", "--minimum-atoms", "7", "--component", "gamma_xy"])
    request = workflow.REQUEST_BUILDERS[args.command](args)
    routes = get_registered_analysis_commands()

    assert args.command == "get_z_binned_deformation_gradient_strain"
    assert isinstance(request, ZBinnedDeformationGradientStrainRequest)
    assert request.minimum_atoms == 7
    assert routes[args.command].module_path == "reaxkit.workflows.stress_strain.z_binned_strain_workflow"


def test_run_main_keeps_sparse_selection_out_of_loader_args(monkeypatch) -> None:
    parser = workflow.build_parser(
        argparse.ArgumentParser(),
        command="get_z_binned_top_bottom_strain",
    )
    args = parser.parse_args(["--z-bins", "2", "--frames", "0", "10", "20", "--no-unwrap"])
    captured = {}

    def fake_run(_self, _task, request, runtime_args):
        captured["request"] = request
        captured["runtime_args"] = runtime_args
        return SimpleNamespace(table=None)

    monkeypatch.setattr(workflow.AnalysisExecutor, "run", fake_run)
    monkeypatch.setattr(workflow, "present_result", lambda *args, **kwargs: None)

    assert workflow.run_main(args.command, args) == 0
    assert captured["request"].selected_frames == [0, 10, 20]
    assert captured["runtime_args"]["frames"] is None


def test_run_main_generates_batch_plots_without_replacing_csv_presentation(monkeypatch, tmp_path) -> None:
    parser = workflow.build_parser(
        argparse.ArgumentParser(),
        command="get_z_binned_top_bottom_strain",
    )
    args = parser.parse_args([
        "--z-bins", "2", "--no-unwrap", "--gen-plots",
        "--output-dir", str(tmp_path), "--dpi", "72",
    ])
    calls = []
    table = pd.DataFrame({"frame": [0]})

    monkeypatch.setattr(
        workflow.AnalysisExecutor,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(table=table),
    )
    monkeypatch.setattr(workflow, "present_result", lambda *args, **kwargs: calls.append("csv"))
    monkeypatch.setattr(
        workflow,
        "generate_top_bottom_plots",
        lambda received, root, **kwargs: calls.append((received, root, kwargs)) or [tmp_path / "plot.png"],
    )

    assert workflow.run_main(args.command, args) == 0
    assert calls[0] == "csv"
    assert calls[1][0] is table
    assert calls[1][1] == tmp_path.resolve()
    assert calls[1][2] == {"dpi": 72, "plot_every": 1, "y_scale": "global"}
