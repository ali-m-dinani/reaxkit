from __future__ import annotations

import argparse
from importlib import import_module, resources as ir

import yaml

from reaxkit.core.registry.analysis_cli_routing_registry import get_registered_analysis_commands
from reaxkit.core.resolve.command_alias_resolver import resolve_command_name
from reaxkit.workflows.timeseries import ALL_COMMANDS


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
