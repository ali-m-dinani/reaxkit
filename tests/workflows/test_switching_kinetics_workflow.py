import argparse

import numpy as np
import pandas as pd
import pytest

from reaxkit.analysis.ferroelectrics.switching_kinetics import kai_fraction
from reaxkit.core.registry.analysis_cli_routing_registry import (
    get_registered_analysis_commands,
)
from reaxkit.workflows.ferroelectrics.switching_kinetics import common
from reaxkit.workflows.ferroelectrics.switching_kinetics import comparison_workflow
from reaxkit.workflows.ferroelectrics.switching_kinetics import kai_workflow


def _parser():
    return comparison_workflow.build_parser(
        argparse.ArgumentParser(), command="fit-switching-kinetics"
    )


def test_prepare_fraction_from_polarization_and_domain_polarities():
    polarization = pd.DataFrame({"t": [10, 11, 12, 13, 14], "P": [-2, -1, 0, 1, 2]})
    normalized = common.prepare_switching_fraction(
        polarization,
        time_column="t",
        data_kind="polarization",
        value_column="P",
        final_polarization=2.0,
    )
    np.testing.assert_allclose(normalized["time"], [0, 1, 2, 3, 4])
    np.testing.assert_allclose(normalized["fraction"], [0, 0.25, 0.5, 0.75, 1])

    polarity = pd.DataFrame(
        {"t": [0, 1, 2, 3, 4], "a": [1, -1, -1, -1, -1], "b": [-1, -1, 1, 1, 1]}
    )
    flipped = common.prepare_switching_fraction(
        polarity,
        time_column="t",
        data_kind="polarity-columns",
        polarity_columns=["a", "b"],
    )
    np.testing.assert_allclose(flipped["fraction"], [0, 0.5, 1, 1, 1])


def test_comparison_workflow_reads_csv_and_writes_ranked_excel(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "switching.csv"
    time = np.linspace(0.0, 8.0, 41)
    pd.DataFrame({"time": time, "fraction": kai_fraction(time, 2.0, 1.5)}).to_csv(
        source, index=False
    )
    args = _parser().parse_args(
        [
            "--input", str(source),
            "--data-kind", "fraction",
            "--value-column", "fraction",
            "--starts", "3",
            "--output", "fits.xlsx",
        ]
    )

    assert comparison_workflow.run_main("fit-switching-kinetics", args) == 0
    run_directory = next(
        (tmp_path / "reaxkit_workspace" / "other" / "fit-switching-kinetics").glob(
            "run_*"
        )
    )
    output = run_directory / "fits.xlsx"
    assert set(pd.ExcelFile(output).sheet_names) == {
        "normalized_data",
        "fitted_curves",
        "parameters",
        "metrics",
        "sweep_curves",
        "sweep_parameters",
        "sweep_metrics",
    }
    metrics = pd.read_excel(output, sheet_name="metrics")
    assert metrics.loc[0, "model"] == "kai"
    assert metrics.loc[0, "rank_by_aicc"] == 1
    assert len(list((run_directory / "plots").glob("*.png"))) == 7


def test_switching_commands_are_registered():
    registry = get_registered_analysis_commands()
    for command in (
        "fit-switching-kinetics",
        "fit-kai-switching",
        "fit-nls-switching",
        "fit-snng-switching",
    ):
        assert command in registry


def test_out_of_range_normalized_polarization_is_explicit():
    data = pd.DataFrame({"time": range(5), "P": [0, 1, 2, 3, 4]})
    with pytest.raises(ValueError, match="outside"):
        common.prepare_switching_fraction(
            data,
            time_column="time",
            data_kind="polarization",
            value_column="P",
            final_polarization=2,
        )


def test_normalization_sorts_before_selecting_polarization_endpoints():
    data = pd.DataFrame({"time": [4, 0, 3, 1, 2], "P": [2, -2, 1, -1, 0]})
    normalized = common.prepare_switching_fraction(
        data,
        time_column="time",
        data_kind="polarization",
        value_column="P",
        final_polarization=2,
    )
    np.testing.assert_allclose(normalized["fraction"], [0, 0.25, 0.5, 0.75, 1])


def test_kai_sweep_refits_each_n_and_writes_blue_red_overlay_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "kai.csv"
    time = np.linspace(0.0, 8.0, 31)
    pd.DataFrame({"time": time, "fraction": kai_fraction(time, 2.0, 2.2)}).to_csv(
        source, index=False
    )
    parser = kai_workflow.build_parser(
        argparse.ArgumentParser(), command="fit-kai-switching"
    )
    args = parser.parse_args(
        [
            "--input", str(source),
            "--data-kind", "fraction",
            "--value-column", "fraction",
            "--sweep", "n=1:3",
            "--fixed", "t0",
            "--time-unit", "ps",
            "--starts", "2",
            "--output", "kai_results.xlsx",
        ]
    )

    assert kai_workflow.run_main("fit-kai-switching", args) == 0
    run_directory = next(
        (tmp_path / "reaxkit_workspace" / "other" / "fit-kai-switching").glob(
            "run_*"
        )
    )
    output = run_directory / "kai_results.xlsx"
    sweep = pd.read_excel(output, sheet_name="sweep_curves")
    assert sorted(sweep["sweep_value"].unique().tolist()) == [1, 2, 3]
    assert set(sweep["sweep_parameter"]) == {"n"}
    sweep_parameters = pd.read_excel(output, sheet_name="sweep_parameters")
    fixed_t0 = sweep_parameters[sweep_parameters["parameter"] == "t0"]
    assert fixed_t0["fixed"].all()
    assert fixed_t0["value"].nunique() == 1
    assert fixed_t0["value"].iloc[0] == pytest.approx(2.0, rel=5e-3)
    assert (run_directory / "plots" / "all_kai_n_sweep.png").is_file()
    assert (run_directory / "plots" / "all_kai_residuals.png").is_file()


def test_sweep_range_is_inclusive_and_accepts_explicit_steps():
    assert common._parse_sweep_values("1:6") == (1, 2, 3, 4, 5, 6)
    assert common._parse_sweep_values("0.5:1.5:0.5") == (0.5, 1.0, 1.5)


def test_default_output_uses_workspace_other_command_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "switching.csv"
    time = np.linspace(0.0, 6.0, 21)
    pd.DataFrame({"time": time, "fraction": kai_fraction(time, 2.0, 1.5)}).to_csv(
        source, index=False
    )
    parser = kai_workflow.build_parser(
        argparse.ArgumentParser(), command="fit-kai-switching"
    )
    args = parser.parse_args(
        [
            "--input", str(source),
            "--data-kind", "fraction",
            "--value-column", "fraction",
            "--starts", "2",
        ]
    )

    assert kai_workflow.run_main("fit-kai-switching", args) == 0
    command_dir = tmp_path / "reaxkit_workspace" / "other" / "fit-kai-switching"
    first_run = next(command_dir.glob("run_*"))
    assert (first_run / "switching_switching_fits.xlsx").is_file()
    assert (first_run / "plots" / "all_kai_best_fit.png").is_file()

    assert kai_workflow.run_main("fit-kai-switching", args) == 0
    run_directories = sorted(command_dir.glob("run_*"))
    assert len(run_directories) == 2
    assert run_directories[0] != run_directories[1]
