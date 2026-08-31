from __future__ import annotations

import argparse
from importlib import import_module, resources as ir
from pathlib import Path
from types import SimpleNamespace

import yaml

import pandas as pd
import pytest

from reaxkit.core.registry.analysis_cli_routing_registry import get_registered_analysis_commands
from reaxkit.core.resolve.command_alias_resolver import resolve_command_name
from reaxkit.workflows.timeseries import ALL_COMMANDS
from reaxkit.workflows.timeseries import common


SCALAR_FIELDS = {
    "get_potential_energy": "potential_energy",
    "get_num_of_atoms": "num_of_atoms",
    "get_volume": "volume",
    "get_temperature": "temperature",
    "get_pressure": "pressure",
    "get_density": "density",
    "get_elapsed_time": "elapsed_time",
    "get_a": "a",
    "get_b": "b",
    "get_c": "c",
    "get_alpha": "alpha",
    "get_beta": "beta",
    "get_gamma": "gamma",
}


def _parser_for(command: str) -> argparse.ArgumentParser:
    module = import_module(f"reaxkit.workflows.timeseries.{command}")
    parser = argparse.ArgumentParser()
    module.build_parser(parser, command=command)
    return parser


def test_every_dedicated_command_has_a_registered_module_and_hyphen_alias() -> None:
    routes = get_registered_analysis_commands()
    for command in ALL_COMMANDS:
        assert routes[command].module_path == f"reaxkit.workflows.timeseries.{command}"
        assert resolve_command_name(command.replace("_", "-"), task_names=routes) == command
        module = import_module(routes[command].module_path)
        assert callable(module.build_parser)
        assert callable(module.build_request)
        assert callable(module.run_main)


def test_scalar_getters_pin_their_supported_field() -> None:
    for command, field in SCALAR_FIELDS.items():
        module = import_module(f"reaxkit.workflows.timeseries.{command}")
        args = _parser_for(command).parse_args(["--frames", "0:10:2", "--every", "2"])
        request = module.build_request(args)
        assert request.field == field
        assert request.frames == [0, 2, 4, 6, 8]
        assert request.every == 2


def test_family_getters_build_requests_without_field_expressions() -> None:
    cases = {
        "get_trajectory": (["--atom-ids", "1", "2", "--dims", "z"], {"atom_ids": (1, 2), "dims": ("z",)}),
        "get_displacement": (["--atom-ids", "3", "--dims", "xy", "--reference-frame", "5"], {"atom_ids": (3,), "dims": ("xy",), "reference_frame": 5}),
        "get_charge": (["--atom-ids", "1", "4"], {"atom_ids": (1, 4)}),
        "get_cell_dimensions": (["--fields", "a", "gamma"], {"fields": ("a", "gamma")}),
        "get_electric_field": (["--components", "field_z"], {"components": ("field_z",)}),
        "get_eregime": (["--field", "field"], {"field": "field"}),
        "get_partial_energy": (["--components", "ebond", "ecoul"], {"components": ("ebond", "ecoul")}),
        "get_restraint": (["--restraint-index", "1", "2"], {"restraint_index": (1, 2)}),
        "get_molecular_frequency": (["--molecules", "H2O", "OH"], {"molecules": ("H2O", "OH")}),
        "get_molecular_totals": (["--quantities", "total_atoms"], {"quantities": ("total_atoms",)}),
        "get_geometry_optimization": (["--components", "E_pot", "RMSG"], {"component": ("E_pot", "RMSG")}),
    }
    for command, (argv, expected) in cases.items():
        module = import_module(f"reaxkit.workflows.timeseries.{command}")
        request = module.build_request(_parser_for(command).parse_args(argv))
        for name, value in expected.items():
            assert getattr(request, name) == value


def test_get_electric_field_accepts_copy_to_dot() -> None:
    args = _parser_for("get_electric_field").parse_args(
        ["--components", "field_z", "--copy-to-dot"]
    )

    assert args.copy_to_dot is True


def test_electric_field_frame_axis_uses_sibling_control_iout2(tmp_path) -> None:
    fort78 = tmp_path / "fort.78"
    fort78.write_text("", encoding="utf-8")
    (tmp_path / "control").write_text("# MD\n5 iout2\n", encoding="utf-8")
    args = argparse.Namespace(
        xaxis="frame",
        control="control",
        fort78=str(fort78),
    )
    table = pd.DataFrame(
        {"frame_index": [0, 1, 2, 3], "iter": [0, 4, 5, 10]}
    )

    values, label, source = common._plot_axis(table, args)

    assert values.tolist() == [0, 0.8, 1, 2]
    assert label == "Frame"
    assert source == "iter"


def test_electric_field_frame_axis_uses_sibling_xmolout_without_control(tmp_path) -> None:
    fort78 = tmp_path / "fort.78"
    fort78.write_text("", encoding="utf-8")
    (tmp_path / "xmolout").write_text(
        "1\nslab 10 -1 1 1 1 90 90 90\nAl 0 0 0\n"
        "1\nslab 20 -2 1 1 1 90 90 90\nAl 0 0 0\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        xaxis="frame",
        control="control",
        fort78=str(fort78),
        frame_source=None,
        frame_count=None,
    )
    table = pd.DataFrame({"frame_index": [0, 1, 2], "iter": [10, 15, 20]})

    values, label, source = common._plot_axis(table, args)

    assert values.tolist() == [0, 0.5, 1]
    assert label == "Frame"
    assert source == "iter"


def test_summary_backed_frame_axis_automatically_uses_sibling_xmolout(tmp_path) -> None:
    summary = tmp_path / "summary.txt"
    summary.write_text("", encoding="utf-8")
    xmolout = tmp_path / "xmolout"
    xmolout.write_text(
        "1\nslab 10 -1 1 1 1 90 90 90\nAl 0 0 0\n"
        "1\nslab 20 -2 1 1 1 90 90 90\nAl 0 0 0\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        xaxis="frame",
        control="control",
        summary=str(summary),
        xmolout="xmolout",
        frame_source=None,
        frame_count=None,
    )
    table = pd.DataFrame({"frame_index": [0, 1, 2], "iter": [10, 15, 20]})

    values, label, source = common._plot_axis(table, args)

    assert values.tolist() == [0, 0.5, 1]
    assert label == "Frame"
    assert source == "iter"
    assert common._axis_frame_source(args) == str(xmolout)


def test_summary_backed_frame_axis_prefers_sibling_control_over_xmolout(tmp_path) -> None:
    summary = tmp_path / "summary.txt"
    summary.write_text("", encoding="utf-8")
    (tmp_path / "control").write_text("# MD\n5 iout2\n", encoding="utf-8")
    (tmp_path / "xmolout").write_text(
        "1\nslab 10 -1 1 1 1 90 90 90\nAl 0 0 0\n"
        "1\nslab 20 -2 1 1 1 90 90 90\nAl 0 0 0\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        xaxis="frame",
        control="control",
        summary=str(summary),
        xmolout="xmolout",
        frame_source=None,
        frame_count=None,
    )
    table = pd.DataFrame({"frame_index": [0, 1, 2], "iter": [0, 5, 10]})

    values, label, source = common._plot_axis(table, args)

    assert values.tolist() == [0, 1, 2]
    assert label == "Frame"
    assert source == "iter"


def test_electric_field_run_task_updates_persisted_frame_index(monkeypatch, tmp_path) -> None:
    fort78 = tmp_path / "fort.78"
    fort78.write_text("", encoding="utf-8")
    (tmp_path / "control").write_text("# MD\n5 iout2\n", encoding="utf-8")
    result = SimpleNamespace(
        table=pd.DataFrame(
            {
                "frame_index": [0, 1, 2, 3],
                "iter": [0, 4, 5, 10],
                "component": ["field_z"] * 4,
                "value": [0.0, 0.1, 0.2, 0.3],
            }
        )
    )
    captured = {}
    monkeypatch.setattr(common.AnalysisExecutor, "run", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(
        common,
        "present_result",
        lambda _command, presented, _args, **_kwargs: captured.setdefault(
            "table", presented.table.copy()
        ),
    )
    args = argparse.Namespace(
        xaxis="frame",
        control="control",
        fort78=str(fort78),
        frame_source=None,
        frame_count=None,
    )

    assert common.run_task("get_electric_field", "electric_field_series", object(), args) == 0

    assert captured["table"]["frame_index"].tolist() == [0, 0.8, 1, 2]
    assert result.csv_tables["integer_frames"]["frame_index"].tolist() == [0, 1, 2]
    assert result.csv_tables["integer_frames"]["iter"].tolist() == [0, 5, 10]


def test_electric_field_run_main_writes_corrected_frame_index_to_csv(
    monkeypatch, tmp_path
) -> None:
    module = import_module("reaxkit.workflows.timeseries.get_electric_field")
    fort78 = tmp_path / "fort.78"
    fort78.write_text("", encoding="utf-8")
    (tmp_path / "control").write_text("# MD\n5 iout2\n", encoding="utf-8")
    result = SimpleNamespace(
        table=pd.DataFrame(
            {
                "frame_index": [0, 1, 2, 3],
                "iter": [0, 4, 5, 10],
                "component": ["field_z"] * 4,
                "value": [0.0, 0.1, 0.2, 0.3],
            }
        )
    )
    monkeypatch.setattr(common.AnalysisExecutor, "run", lambda *_args, **_kwargs: result)
    export_path = tmp_path / "electric-field.csv"
    args = _parser_for("get_electric_field").parse_args(
        [
            "--components",
            "field_z",
            "--fort78",
            str(fort78),
            "--xaxis",
            "frame",
            "--export",
            str(export_path),
            "--project-root",
            str(tmp_path / "workspace"),
            "--run-id",
            "frame-csv-test",
        ]
    )

    assert module.run_main("get_electric_field", args) == 0

    exported = pd.read_csv(export_path)
    assert exported["frame_index"].tolist() == [0, 0.8, 1, 2]
    integer_export = pd.read_csv(tmp_path / "electric-field_integer_frames.csv")
    assert integer_export["frame_index"].tolist() == [0, 1, 2]
    assert integer_export["iter"].tolist() == [0, 5, 10]


def test_electric_field_saved_plot_run_writes_corrected_automatic_result_csv(
    monkeypatch, tmp_path
) -> None:
    module = import_module("reaxkit.workflows.timeseries.get_electric_field")
    fort78 = tmp_path / "fort.78"
    fort78.write_text("", encoding="utf-8")
    (tmp_path / "control").write_text("# MD\n5 iout2\n", encoding="utf-8")
    result = SimpleNamespace(
        table=pd.DataFrame(
            {
                "frame_index": [0, 1, 2, 3],
                "iter": [0, 4, 5, 10],
                "component": ["field_z"] * 4,
                "value": [0.0, 0.1, 0.2, 0.3],
            }
        )
    )
    monkeypatch.setattr(common.AnalysisExecutor, "run", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(
        "reaxkit.presentation.dispatcher.render_plot",
        lambda payload: Path(payload["save"]).write_bytes(b"plot"),
    )
    workspace = tmp_path / "workspace"
    args = _parser_for("get_electric_field").parse_args(
        [
            "--components",
            "field_z",
            "--fort78",
            str(fort78),
            "--xaxis",
            "frame",
            "--plot",
            "single",
            "--save",
            str(tmp_path / "field.png"),
            "--project-root",
            str(workspace),
            "--run-id",
            "automatic-csv-test",
        ]
    )

    assert module.run_main("get_electric_field", args) == 0

    automatic_dir = (
        workspace
        / "analysis"
        / "get_electric_field"
        / "automatic-csv-test"
    )
    all_frames = pd.read_csv(automatic_dir / "all_frames.csv")
    integer_frames = pd.read_csv(automatic_dir / "integer_frames.csv")
    assert all_frames["frame_index"].tolist() == [0, 0.8, 1, 2]
    assert integer_frames["frame_index"].tolist() == [0, 1, 2]
    assert integer_frames["iter"].tolist() == [0, 5, 10]


def test_electric_field_frame_axis_errors_before_persistence_without_source(
    monkeypatch, tmp_path
) -> None:
    fort78 = tmp_path / "fort.78"
    fort78.write_text("", encoding="utf-8")
    result = SimpleNamespace(
        table=pd.DataFrame(
            {
                "frame_index": [0, 1],
                "iter": [0, 5],
                "component": ["field_z", "field_z"],
                "value": [0.0, 0.1],
            }
        )
    )
    monkeypatch.setattr(common.AnalysisExecutor, "run", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(
        common,
        "present_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("presentation must not run without a frame source")
        ),
    )
    args = argparse.Namespace(
        xaxis="frame",
        control="control",
        fort78=str(fort78),
        frame_source=None,
        frame_count=None,
    )

    with pytest.raises(FileNotFoundError, match="Pass --frame-count"):
        common.run_task("get_electric_field", "electric_field_series", object(), args)


def test_get_frames_count_accepts_general_and_engine_specific_paths() -> None:
    module = import_module("reaxkit.workflows.timeseries.get_frames_count")
    default_args = _parser_for("get_frames_count").parse_args([])
    module._normalize_trajectory_source(default_args)
    assert default_args.input == "."

    for argv, expected in (
        (["trajectory.dat"], "trajectory.dat"),
        (["--input", "trajectory.dat"], "trajectory.dat"),
        (["--file", "trajectory.dat"], "trajectory.dat"),
    ):
        args = _parser_for("get_frames_count").parse_args(argv)
        module._normalize_trajectory_source(args)
        assert args.input == expected
        assert args.xmolout == expected
        assert args.dump == expected
        assert args.rkf == expected

    args = _parser_for("get_frames_count").parse_args(["--engine", "lammps", "--dump", "dump.lammpstrj"])
    module._normalize_trajectory_source(args)
    assert args.input == "dump.lammpstrj"
    assert args.dump == "dump.lammpstrj"


def test_get_frames_count_uses_fast_engine_probe_without_loading_trajectory(monkeypatch, capsys) -> None:
    module = import_module("reaxkit.workflows.timeseries.get_frames_count")

    class FastAdapter:
        @staticmethod
        def quick_n_frames(runtime_args):
            assert runtime_args["input"] == "trajectory.dat"
            return 7

    monkeypatch.setattr(module, "resolve_engine", lambda *_args, **_kwargs: FastAdapter())
    monkeypatch.setattr(
        module.AnalysisExecutor,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("TrajectoryData should not be loaded")),
    )
    args = _parser_for("get_frames_count").parse_args(["trajectory.dat"])

    assert module.run_main("get_frames_count", args) == 0
    assert capsys.readouterr().out == "7\n"


def test_get_frames_count_falls_back_to_trajectory_analysis(monkeypatch, capsys) -> None:
    module = import_module("reaxkit.workflows.timeseries.get_frames_count")

    class NoFastCountAdapter:
        @staticmethod
        def quick_n_frames(_runtime_args):
            return None

    def fake_run(_self, task, request, runtime_args):
        assert task.required_data.__name__ == "TrajectoryData"
        assert runtime_args["input"] == "trajectory.dat"
        return module.FramesCountResult(count=9, request=request)

    monkeypatch.setattr(module, "resolve_engine", lambda *_args, **_kwargs: NoFastCountAdapter())
    monkeypatch.setattr(module.AnalysisExecutor, "run", fake_run)
    args = _parser_for("get_frames_count").parse_args(["trajectory.dat"])

    assert module.run_main("get_frames_count", args) == 0
    assert capsys.readouterr().out == "9\n"


def test_dedicated_commands_are_in_workflow_metadata() -> None:
    resource = ir.files("reaxkit.workflows.data").joinpath("workflow_dataclass_map.yaml")
    with resource.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    routes = document["command_to_workflow_module"]
    workflows = document["workflows"]
    for command in ALL_COMMANDS:
        assert routes[command] == f"reaxkit.workflows.timeseries.{command}"
        assert routes[command.replace("_", "-")] == routes[command]
        assert workflows[command]["parent_workflow"] == "timeseries_workflow"
