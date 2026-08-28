from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from reaxkit.analysis.force_field.diagnostic_beeswarm import (
    FFieldOptimizationDiagnosticBeeswarmRequest,
    FFieldOptimizationDiagnosticBeeswarmTask,
    build_diagnostic_beeswarm_tables,
    normalize_evolution_values,
)
from reaxkit.domain.data_models import (
    ForceFieldOptimizationDiagnosticData,
    ForceFieldOptimizationDiagnosticPlotData,
    ForceFieldOptimizationParameterData,
    ForceFieldParametersData,
)
from reaxkit.presentation.plot.renderers.beeswarm import BeeswarmPlotRenderer
from reaxkit.presentation.plot.renderers.diagnostic_evolution import DiagnosticEvolutionPlotRenderer
from reaxkit.workflows.file_tools.ffield_workflow import (
    _build_parameter_optimization_diagnostic_beeswarm_request,
    _build_sensitivity_beeswarm_result,
    _build_tornado_result,
    _plot_payload,
    build_parser,
)


def _plot_data() -> ForceFieldOptimizationDiagnosticPlotData:
    diagnostics = ForceFieldOptimizationDiagnosticData(
        identifiers=np.array(["3 1 1", "2 1 1"], dtype=object),
        value1=np.array([0.0, 1.0]),
        value2=np.array([5.0, 1.5]),
        value3=np.array([10.0, 2.0]),
        diff1=np.array([10.0, 100.0]),
        diff2=np.array([5.0, 150.0]),
        diff3=np.array([8.0, 125.0]),
        a=np.array([1.0, 1.0]),
        b=np.array([1.0, 1.0]),
        c=np.array([1.0, 1.0]),
        parabol_min=np.array([7.5, 1.25]),
        parabol_min_diff=np.array([3.0, 110.0]),
        value4=np.array([8.0, 1.75]),
        diff4=np.array([1.0, 200.0]),
    )
    force_field = ForceFieldParametersData(
        general_parameters=pd.DataFrame(),
        atom_parameters=pd.DataFrame(
            [
                {"symbol": "C", "cov.r": 1.4},
                {"symbol": "H", "cov.r": 0.8},
            ],
            index=[1, 2],
        ),
        bond_parameters=pd.DataFrame([{"i": 1, "j": 2, "Edis1": 5.0}], index=[1]),
        off_diagonal_parameters=pd.DataFrame(),
        angle_parameters=pd.DataFrame(),
        torsion_parameters=pd.DataFrame(),
        hydrogen_bond_parameters=pd.DataFrame(),
    )
    optimization_parameters = ForceFieldOptimizationParameterData(
        ff_section=np.array([3, 2]),
        ff_section_line=np.array([1, 1]),
        ff_parameter=np.array([1, 1]),
        search_interval=np.array([0.25, 0.05]),
        min_value=np.array([0.0, 1.0]),
        max_value=np.array([10.0, 2.0]),
        inline_comment=np.array(["bond", "atom"], dtype=object),
    )
    return ForceFieldOptimizationDiagnosticPlotData(
        diagnostics=diagnostics,
        force_field_parameters=force_field,
        optimization_parameters=optimization_parameters,
    )


def test_analysis_normalizes_with_declared_bounds_and_resolves_parameter_labels() -> None:
    samples, parameters, excluded = build_diagnostic_beeswarm_tables(_plot_data())

    assert excluded == 0
    assert parameters["parameter_key"].tolist() == ["2 1 1", "3 1 1"]
    assert parameters["parameter_label"].tolist() == ["Atom C - cov.r", "Bond C-H - Edis1"]
    bond_samples = samples.loc[samples["parameter_key"] == "3 1 1"]
    assert bond_samples["normalized_value"].tolist() == pytest.approx([0.0, 0.5, 1.0, 0.75, 0.8])
    bond = parameters.loc[parameters["parameter_key"] == "3 1 1"].iloc[0]
    assert bond["lower_bound"] == 0.0
    assert bond["upper_bound"] == 10.0
    assert bond["search_interval"] == 0.25
    assert bond["starting_value"] == 5.0
    assert bond["final_value"] == 8.0
    assert (bond["color_min"], bond["color_max"]) == (1.0, 10.0)


def test_analysis_supports_global_objective_scale_and_value_sorting() -> None:
    request = FFieldOptimizationDiagnosticBeeswarmRequest(
        sort_by="final",
        global_objective_scale=True,
    )
    result = FFieldOptimizationDiagnosticBeeswarmTask().run(_plot_data(), request)

    assert result.request == request
    assert result.parameters["parameter_key"].tolist() == ["2 1 1", "3 1 1"]
    assert set(result.parameters["color_min"]) == {1.0}
    assert set(result.parameters["color_max"]) == {200.0}
    assert set(result.table["plot_row"]) == {0, 1}

    _, starting, _ = build_diagnostic_beeswarm_tables(_plot_data(), sort_by="starting")
    assert starting["parameter_key"].tolist() == ["2 1 1", "3 1 1"]


def test_analysis_excludes_parameters_without_optimization_bounds() -> None:
    data = _plot_data()
    data.optimization_parameters.min_value[0] = np.nan

    samples, parameters, excluded = build_diagnostic_beeswarm_tables(data)

    assert excluded == 1
    assert parameters["parameter_key"].tolist() == ["2 1 1"]
    assert set(samples["parameter_key"]) == {"2 1 1"}


def test_analysis_accepts_descending_stored_bounds_and_top_filter() -> None:
    data = _plot_data()
    data.optimization_parameters.min_value[0] = 10.0
    data.optimization_parameters.max_value[0] = 0.0

    samples, parameters, excluded = build_diagnostic_beeswarm_tables(data, top=1)

    assert excluded == 0
    assert parameters["parameter_key"].tolist() == ["2 1 1"]
    assert set(samples["parameter_key"]) == {"2 1 1"}

    bond_samples, _, _ = build_diagnostic_beeswarm_tables(data)
    bond = bond_samples.loc[bond_samples["parameter_key"] == "3 1 1"]
    assert bond["normalized_value"].tolist() == pytest.approx([1.0, 0.5, 0.0, 0.25, 0.2])


def test_workflow_builds_bounded_diagnostic_plot_payload() -> None:
    parser = build_parser(argparse.ArgumentParser(), command="get_ffield_diagnostic_data")
    args = parser.parse_args([
        "--plot",
        "beeswarm",
        "--sort",
        "final",
        "--global-objective-scale",
        "--params",
        "custom.params",
    ])
    request = _build_parameter_optimization_diagnostic_beeswarm_request(args)
    result = FFieldOptimizationDiagnosticBeeswarmTask().run(_plot_data(), request)
    payload = _plot_payload("get_ffield_diagnostic_data", result, args)

    assert args.params == "custom.params"
    assert request.sort_by == "final"
    assert request.global_objective_scale is True
    assert payload is not None
    assert payload["plot_type"] == "beeswarm_plot"
    assert payload["global_objective_scale"] is True
    assert payload["x"] == pytest.approx(result.table["normalized_value"].tolist())
    assert len(payload["diagnostic_parameters"]) == 2


def test_dedicated_diagnostic_commands_expose_separate_plot_contracts() -> None:
    sensitivity_parser = build_parser(
        argparse.ArgumentParser(),
        command="get_ffield_diagnostics_sensitivity",
    )
    sensitivity_args = sensitivity_parser.parse_args([])
    assert sensitivity_args.plot == "tornado"
    assert sensitivity_parser.parse_args(["--plot", "beeswarm"]).plot == "beeswarm"
    with pytest.raises(SystemExit):
        sensitivity_parser.parse_args(["--plot", "single"])

    evolution_parser = build_parser(
        argparse.ArgumentParser(),
        command="get_ffield_diagnostics_evolution",
    )
    evolution_args = evolution_parser.parse_args([])
    assert evolution_args.plot == "beeswarm"
    assert evolution_args.normalization == "bound_min"
    scatter_args = evolution_parser.parse_args(["--plot", "scatter", "--normalization", "first"])
    assert scatter_args.plot == "scatter"
    assert scatter_args.normalization == "first"
    with pytest.raises(SystemExit):
        evolution_parser.parse_args(["--plot", "tornado"])


def test_sensitivity_command_builds_beeswarm_and_tornado_views() -> None:
    base_result = argparse.Namespace(
        table=pd.DataFrame(
            {
                "identifier": ["3 1 1", "3 1 1", "2 1 1"],
                "value1": [1.0, 1.2, 2.0],
                "value2": [1.5, 1.6, 2.2],
                "value4": [1.8, 1.9, 2.4],
                "sensitivity1/3": [0.90, 0.92, 0.99],
                "sensitivity2/3": [1.00, 1.02, 1.00],
                "sensitivity4/3": [1.10, 1.08, 1.01],
                "min_sensitivity": [0.90, 0.92, 0.99],
                "max_sensitivity": [1.10, 1.08, 1.01],
            }
        )
    )
    beeswarm = _build_sensitivity_beeswarm_result(base_result, top=1)
    beeswarm_args = argparse.Namespace(plot="beeswarm", vline=1.0)
    beeswarm_payload = _plot_payload(
        "get_ffield_diagnostics_sensitivity",
        beeswarm,
        beeswarm_args,
    )

    assert set(beeswarm.table["parameter"]) == {"3 1 1"}
    assert len(beeswarm.table) == 6
    assert beeswarm_payload is not None
    assert beeswarm_payload["plot_type"] == "beeswarm_plot"
    assert beeswarm_payload["xlabel"] == "Sensitivity (Error Response)"
    assert beeswarm_payload["vline"] == 1.0

    tornado = _build_tornado_result(base_result, top=1)
    tornado_payload = _plot_payload(
        "get_ffield_diagnostics_sensitivity",
        tornado,
        argparse.Namespace(plot="tornado", vline=1.0),
    )
    assert tornado.table["parameter"].tolist() == ["3 1 1"]
    assert tornado_payload is not None
    assert tornado_payload["plot_type"] == "tornado_plot"


def test_evolution_command_uses_bounded_diagnostic_payload() -> None:
    args = build_parser(
        argparse.ArgumentParser(),
        command="get_ffield_diagnostics_evolution",
    ).parse_args(["--sort", "final"])
    result = FFieldOptimizationDiagnosticBeeswarmTask().run(
        _plot_data(),
        _build_parameter_optimization_diagnostic_beeswarm_request(args),
    )

    payload = _plot_payload("get_ffield_diagnostics_evolution", result, args)

    assert payload is not None
    assert payload["plot_type"] == "beeswarm_plot"
    assert payload["xlabel"] == "Normalized parameter value within optimization bounds"
    assert payload["x"] == pytest.approx(result.table["normalized_value"].tolist())


def test_evolution_scatter_uses_epochs_and_selected_normalization() -> None:
    args = build_parser(
        argparse.ArgumentParser(),
        command="get_ffield_diagnostics_evolution",
    ).parse_args(["--plot", "scatter", "--normalization", "first"])
    result = FFieldOptimizationDiagnosticBeeswarmTask().run(
        _plot_data(),
        _build_parameter_optimization_diagnostic_beeswarm_request(args),
    )

    payload = _plot_payload("get_ffield_diagnostics_evolution", result, args)

    assert payload is not None
    assert payload["plot_type"] == "diagnostic_evolution_plot"
    assert payload["normalization"] == "first"
    assert payload["x"] == [1, 1]
    assert payload["y"] == pytest.approx([0.0, 0.0])
    assert payload["hue"] == pytest.approx([200.0, 1.0])


def test_evolution_normalization_uses_common_bound_span() -> None:
    values = [2.0, 4.0, 8.0]

    assert normalize_evolution_values(
        values, lower_bound=0.0, upper_bound=10.0, normalization="bound_min"
    ).tolist() == pytest.approx([0.2, 0.4, 0.8])
    assert normalize_evolution_values(
        values, lower_bound=0.0, upper_bound=10.0, normalization="first"
    ).tolist() == pytest.approx([0.0, 0.2, 0.6])


def test_evolution_renderer_builds_one_line_axis_and_colorbar_per_parameter(monkeypatch) -> None:
    args = build_parser(
        argparse.ArgumentParser(),
        command="get_ffield_diagnostics_evolution",
    ).parse_args(["--plot", "scatter"])
    result = FFieldOptimizationDiagnosticBeeswarmTask().run(
        _plot_data(),
        _build_parameter_optimization_diagnostic_beeswarm_request(args),
    )
    payload = _plot_payload("get_ffield_diagnostics_evolution", result, args)
    assert payload is not None
    monkeypatch.setattr(
        "reaxkit.presentation.plot.renderers.diagnostic_evolution.save_or_show",
        lambda figure, _cfg: figure,
    )

    figure = DiagnosticEvolutionPlotRenderer().render(payload)

    assert len(figure.axes) == 4
    assert figure.axes[0].get_ylabel() == "Normalized value"
    assert figure.axes[1].get_xlabel() == "Epoch"
    assert len(figure.axes[0].lines) == 1


def test_sensitivity_renderer_uses_dedicated_matplotlib_path(monkeypatch) -> None:
    payload = {
        "plot_type": "beeswarm_plot",
        "sensitivity_beeswarm": True,
        "x": [0.95, 1.0, 1.05, 0.98, 1.01],
        "y": ["3 1 1", "3 1 1", "3 1 1", "2 1 1", "2 1 1"],
        "hue": [1.0, 2.0, 3.0, 0.2, 0.8],
        "vline": 1.0,
    }
    monkeypatch.setattr(
        "reaxkit.presentation.plot.renderers.beeswarm.save_or_show",
        lambda figure, _cfg: figure,
    )

    figure = BeeswarmPlotRenderer().render(payload)

    axes = figure.axes[0]
    assert [tick.get_text() for tick in axes.get_yticklabels()] == ["3 1 1", "2 1 1"]
    assert axes.get_xlabel() == "Sensitivity (Error Response)"
    assert axes.get_xlim() == pytest.approx((0.9, 1.1))


def test_renderer_reserves_separate_annotation_and_colorbar_regions(monkeypatch) -> None:
    args = argparse.Namespace(plot="beeswarm")
    request = FFieldOptimizationDiagnosticBeeswarmRequest()
    result = FFieldOptimizationDiagnosticBeeswarmTask().run(_plot_data(), request)
    payload = _plot_payload("get_ffield_diagnostic_data", result, args)
    assert payload is not None

    monkeypatch.setattr(
        "reaxkit.presentation.plot.renderers.beeswarm.save_or_show",
        lambda figure, _cfg: figure,
    )
    figure = BeeswarmPlotRenderer().render(payload)

    main_axes = figure.axes[0]
    colorbar_axes = figure.axes[1:]
    annotation_positions = [text.get_position()[0] for text in figure.texts if text.get_text().startswith("[")]
    assert main_axes.get_position().x1 <= 0.55
    assert len(colorbar_axes) == len(result.parameters)
    assert annotation_positions
    assert max(annotation_positions) < min(axes.get_position().x0 for axes in colorbar_axes)
    assert [tick.get_text() for tick in main_axes.get_yticklabels()] == ["2 1 1", "3 1 1"]

    global_request = FFieldOptimizationDiagnosticBeeswarmRequest(global_objective_scale=True)
    global_result = FFieldOptimizationDiagnosticBeeswarmTask().run(_plot_data(), global_request)
    global_payload = _plot_payload("get_ffield_diagnostic_data", global_result, args)
    assert global_payload is not None
    global_figure = BeeswarmPlotRenderer().render(global_payload)
    assert len(global_figure.axes) == 2
