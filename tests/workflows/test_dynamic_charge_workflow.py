from __future__ import annotations

import argparse

import numpy as np

from reaxkit.analysis.ferroelectrics.dynamic_charge import (
    DynamicChargeChangeRequest,
    calculate_dynamic_charge_changes,
)
from reaxkit.core.registry.analysis_cli_routing_registry import get_registered_analysis_commands
from reaxkit.domain.data_models import ChargeData, SimulationData
from reaxkit.workflows.ferroelectrics import dynamic_charge_workflow as workflow


def test_parser_builds_request_and_registers_command() -> None:
    parser = workflow.build_parser(argparse.ArgumentParser(), command=workflow.COMMAND)
    args = parser.parse_args(
        [
            "--atom-numbers", "2", "4",
            "--frames", "0:6:2",
            "--every", "2",
            "--gen-plots",
            "--global-y-axis",
            "--skip-detailed-csv",
            "--x-axis", "iter",
        ]
    )
    request = workflow.build_request(args)

    assert request == DynamicChargeChangeRequest(
        atom_numbers=(2, 4), selected_frames=[0, 2, 4], every=2
    )
    assert args.gen_plots is True
    assert args.global_y_axis is True
    assert args.skip_detailed_csv is True
    assert args.x_axis == "iter"
    route = get_registered_analysis_commands()[workflow.COMMAND]
    assert route.module_path == "reaxkit.workflows.ferroelectrics.dynamic_charge_workflow"


def test_output_dir_writes_all_named_csv_files(tmp_path) -> None:
    data = ChargeData(
        charges=np.array([[0.1], [0.2]]),
        iterations=np.array([0, 1]),
        simulation=SimulationData(atom_ids=[1], elements=["Ti"]),
    )
    result = calculate_dynamic_charge_changes(data, DynamicChargeChangeRequest())
    args = argparse.Namespace(
        output_dir=tmp_path,
        analysis_id=None,
        run_id=None,
        project_root=tmp_path,
    )

    workflow._write_requested_output(result, args)

    assert (tmp_path / "charges.csv").is_file()
    assert (tmp_path / "summary_per_atom.csv").is_file()
    assert (tmp_path / "summary_per_frame_for_all_atoms.csv").is_file()
    assert (tmp_path / "summary_per_frame_per_atom_type.csv").is_file()
    assert not (tmp_path / "summary.csv").exists()


def test_output_dir_skips_detailed_csv_when_requested(tmp_path) -> None:
    data = ChargeData(
        charges=np.array([[0.1], [0.2]]),
        iterations=np.array([0, 1]),
        simulation=SimulationData(atom_ids=[1], elements=["Ti"]),
    )
    result = calculate_dynamic_charge_changes(data, DynamicChargeChangeRequest())
    args = argparse.Namespace(
        output_dir=tmp_path,
        analysis_id=None,
        run_id=None,
        project_root=tmp_path,
        skip_detailed_csv=True,
    )

    workflow._write_requested_output(result, args)

    assert not (tmp_path / "charges.csv").exists()
    assert (tmp_path / "summary_per_atom.csv").is_file()
    assert (tmp_path / "summary_per_frame_for_all_atoms.csv").is_file()
    assert (tmp_path / "summary_per_frame_per_atom_type.csv").is_file()


def test_plot_generation_writes_both_per_atom_plot_families(tmp_path) -> None:
    data = ChargeData(
        charges=np.array([[0.1], [0.2]]),
        iterations=np.array([0, 10]),
        simulation=SimulationData(atom_ids=[7], elements=["Ti"]),
    )
    result = calculate_dynamic_charge_changes(data, DynamicChargeChangeRequest())

    written = workflow.generate_atom_charge_plots(
        result,
        tmp_path,
        x_axis="iter",
        dpi=72,
    )

    assert len(written) == 2
    assert (tmp_path / "plots" / "charges" / "atom_7_Ti.png").is_file()
    assert (tmp_path / "plots" / "delta_charge" / "atom_7_Ti.png").is_file()


def test_global_y_axis_uses_shared_limits_for_each_plot_family(monkeypatch, tmp_path) -> None:
    data = ChargeData(
        charges=np.array([[0.1, -0.4], [0.2, -0.1]]),
        iterations=np.array([0, 10]),
        simulation=SimulationData(atom_ids=[7, 8], elements=["Ti", "O"]),
    )
    result = calculate_dynamic_charge_changes(data, DynamicChargeChangeRequest())
    captured = []

    def fake_plot_pair(*_args, **kwargs):
        captured.append((kwargs["charge_ylim"], kwargs["delta_ylim"]))
        return []

    monkeypatch.setattr(workflow, "_write_atom_plot_pair", fake_plot_pair)

    workflow.generate_atom_charge_plots(
        result,
        tmp_path,
        x_axis="frame",
        progress=False,
        global_y_axis=True,
    )

    assert len(captured) == 2
    assert captured[0] == captured[1]
    assert np.allclose(captured[0][0], (-0.43, 0.23))
    assert np.allclose(captured[0][1], (-0.015, 0.315))


def test_run_main_requests_charge_only_streaming_and_keeps_baseline_available(
        monkeypatch,
        tmp_path,
) -> None:
    parser = workflow.build_parser(argparse.ArgumentParser(), command=workflow.COMMAND)
    args = parser.parse_args(
        ["--frames", "2", "4", "--project-root", str(tmp_path), "--run-id", "run_test"]
    )
    captured = {}
    result = calculate_dynamic_charge_changes(
        ChargeData(
            charges=np.array([[0.1], [0.2]]),
            simulation=SimulationData(atom_ids=[1], elements=["Ti"]),
        ),
        DynamicChargeChangeRequest(),
    )

    def fake_run(_self, _task, request, runtime_args):
        captured["request"] = request
        captured["runtime_args"] = runtime_args
        return result

    monkeypatch.setattr(workflow.AnalysisExecutor, "run", fake_run)
    monkeypatch.setattr(workflow, "present_result", lambda *_args, **_kwargs: None)

    assert workflow.run_main(workflow.COMMAND, args) == 0
    assert captured["request"].selected_frames == [2, 4]
    assert captured["runtime_args"]["frames"] is None
    assert captured["runtime_args"]["_quick_charge_only"] is True
