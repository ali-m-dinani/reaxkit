from __future__ import annotations

import argparse
import io
import json
import sys
from importlib import import_module
from types import SimpleNamespace
from typing import Any, cast

import pytest

from reaxkit import cli_startup

cli_main = import_module("reaxkit.cli.main")


class _SingleWriteLimitedStream(io.StringIO):
    """Simulate a terminal that drops text beyond each write-size limit."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit

    def write(self, value: str) -> int:
        return super().write(value[: self.limit])


def test_cli_help_is_written_in_terminal_safe_chunks():
    parser = cli_main._ReaxKitArgumentParser()
    message = "".join(f"help row {index}\n" for index in range(1000))
    stream = _SingleWriteLimitedStream(limit=256)

    parser._print_message(message, stream)

    assert stream.getvalue() == message


def test_canonicalize_direct_command_alias():
    argv = ["reaxkit", "mean-square-displacement", "--plot"]

    out = cli_main._canonicalize_direct_command(argv)

    assert out[1] == "msd"


def test_canonicalize_direct_command_diffusivity_alias():
    argv = ["reaxkit", "diffusion-coefficient", "--plot"]

    out = cli_main._canonicalize_direct_command(argv)

    assert out[1] == "diffusivity"


def test_cli_announces_command_immediately_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(sys, "argv", ["reaxkit", "--no-stream", "get-dipole"])

    cli_startup.announce_command_start(sys.argv)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "[ReaxKit] Command 'get-dipole' received; starting work...\n"
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["reaxkit"],
        ["reaxkit", "--help"],
        ["reaxkit", "get-dipole", "--help"],
    ],
)
def test_cli_does_not_announce_for_help_only_invocations(
    argv,
    capsys: pytest.CaptureFixture[str],
):
    cli_startup.announce_command_start(argv)

    assert capsys.readouterr().err == ""


def test_cli_bootstrap_announces_before_loading_full_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(sys, "argv", ["reaxkit", "get-dipole"])

    def fake_run_cli():
        assert capsys.readouterr().err == (
            "[ReaxKit] Command 'get-dipole' received; starting work...\n"
        )
        return 0

    monkeypatch.setattr(cli_startup, "_run_cli", fake_run_cli)

    assert cli_startup.main() == 0


@pytest.mark.parametrize("command", ["get-dipole", "get_dipole", "dipole"])
def test_canonicalize_get_dipole_aliases(command: str):
    argv = ["reaxkit", command, "--scope", "total", "--export", "dipole.csv"]

    out = cli_main._canonicalize_direct_command(argv)

    assert out[1] == "get-dipole"


def test_get_dipole_is_the_registered_command_and_task():
    from reaxkit.analysis.electrostatics.electrostatics import DipoleTask
    from reaxkit.core.registry.analysis_cli_routing_registry import (
        get_registered_analysis_commands,
    )
    from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY

    spec = get_registered_analysis_commands()["get-dipole"]

    assert spec.aliases == ("get_dipole", "dipole")
    assert "dipole" not in get_registered_analysis_commands()
    assert TASK_REGISTRY["get-dipole"] is DipoleTask


def test_get_dipole_accepts_frame_option_alias():
    from reaxkit.workflows.electrostatics import electrostatics_workflow

    parser = argparse.ArgumentParser()
    electrostatics_workflow.build_parser(parser, command="get-dipole")

    args = parser.parse_args(["--scope", "total", "--frame", "0:200:1"])

    assert args.frames == ["0:200:1"]


@pytest.mark.parametrize("command", ["get-dipole", "get-polarization"])
def test_electrostatics_commands_accept_formal_charges(command: str):
    from reaxkit.workflows.electrostatics import electrostatics_workflow

    parser = argparse.ArgumentParser()
    electrostatics_workflow.build_parser(parser, command=command)
    args = parser.parse_args(
        ["--charge-source", "formal", "--formal-charge", "Al=3", "N=-3"]
    )
    builder = electrostatics_workflow.REQUEST_BUILDERS[command]

    request = builder(args)

    assert request.charge_source == "formal"
    assert request.formal_charges == {"Al": 3.0, "N": -3.0}


@pytest.mark.parametrize("command", ["get-dipole", "get-polarization"])
@pytest.mark.parametrize("scope", ["total", "local"])
@pytest.mark.parametrize("volume_method", ["hull", "bbox", "cell"])
def test_electrostatics_commands_accept_volume_method(
    command: str, scope: str, volume_method: str
):
    from reaxkit.workflows.electrostatics import electrostatics_workflow

    parser = argparse.ArgumentParser()
    electrostatics_workflow.build_parser(parser, command=command)
    argv = ["--scope", scope, "--volume-method", volume_method]
    if scope == "local":
        argv.extend(["--core", "Al"])

    request = electrostatics_workflow.REQUEST_BUILDERS[command](parser.parse_args(argv))

    assert request.volume_method == volume_method


def test_electrostatics_volume_method_defaults_follow_scope():
    from reaxkit.workflows.electrostatics import electrostatics_workflow

    dipole_parser = argparse.ArgumentParser()
    electrostatics_workflow.build_parser(dipole_parser, command="get-dipole")
    polarization_parser = argparse.ArgumentParser()
    electrostatics_workflow.build_parser(
        polarization_parser, command="get-polarization"
    )

    dipole = electrostatics_workflow._build_dipole_request(
        dipole_parser.parse_args([])
    )
    total = electrostatics_workflow._build_polarization_request(
        polarization_parser.parse_args([])
    )
    local = electrostatics_workflow._build_polarization_request(
        polarization_parser.parse_args(["--scope", "local", "--core", "Al"])
    )

    assert dipole.volume_method is None
    assert total.volume_method == "hull"
    assert local.volume_method == "bbox"


@pytest.mark.parametrize("command", ["get-polarization", "get_polarization", "polarization"])
def test_canonicalize_polarization_aliases(command: str):
    argv = ["reaxkit", command, "--scope", "total"]

    out = cli_main._canonicalize_direct_command(argv)

    assert out[1] == "get-polarization"


def test_get_polarization_is_the_registered_command_and_task():
    from reaxkit.analysis.electrostatics.electrostatics import PolarizationTask
    from reaxkit.core.registry.analysis_cli_routing_registry import (
        get_registered_analysis_commands,
    )
    from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY

    commands = get_registered_analysis_commands()
    spec = commands["get-polarization"]

    assert spec.aliases == ("polarization",)
    assert "polarization" not in commands
    assert TASK_REGISTRY["get-polarization"] is PolarizationTask


@pytest.mark.parametrize("command", ["get_polarization_field", "polarization_field"])
def test_canonicalize_polarization_field_aliases(command: str):
    argv = ["reaxkit", command, "--aggregate", "mean"]

    out = cli_main._canonicalize_direct_command(argv)

    assert out[1] == "get_polarization_field"


def test_get_polarization_field_is_the_registered_command_and_task():
    from reaxkit.analysis.electrostatics.electrostatics import PolarizationFieldTask
    from reaxkit.core.registry.analysis_cli_routing_registry import (
        get_registered_analysis_commands,
    )
    from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY

    spec = get_registered_analysis_commands()["get_polarization_field"]

    assert spec.aliases == ("polarization_field",)
    assert "polarization_field" not in get_registered_analysis_commands()
    assert TASK_REGISTRY["get_polarization_field"] is PolarizationFieldTask


def test_get_polarization_field_accepts_volume_method():
    from reaxkit.workflows.electrostatics import electrostatics_workflow

    parser = argparse.ArgumentParser()
    electrostatics_workflow.build_parser(parser, command="get_polarization_field")

    default_request = electrostatics_workflow._build_polarization_field_request(parser.parse_args([]))
    args = parser.parse_args(["--volume-method", "bbox"])
    request = electrostatics_workflow._build_polarization_field_request(args)

    assert default_request.volume_method == "hull"
    assert request.volume_method == "bbox"


def test_unknown_flag_for_existing_command_has_custom_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["reaxkit", "--log-in-terminal", "msd", "--atom-ids", "1", "--export", "msd_1.png", "--coords", "2"],
    )

    with pytest.raises(SystemExit) as e:
        cli_main.main()
    assert e.value.code == 2

    err = capsys.readouterr().err
    assert "There is no flag --coords for command msd." in err
    assert "Please run reaxkit msd -h" in err


def test_unknown_command_has_custom_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(sys, "argv", ["reaxkit", "--log-in-terminal", "masadeq", "--coords", "2"])

    with pytest.raises(SystemExit) as e:
        cli_main.main()
    assert e.value.code == 2

    err = capsys.readouterr().err
    assert "There is no command masadeq." in err
    assert 'Please run reaxkit help "query"' in err


def test_successful_cli_command_writes_human_readable_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    class FakeWorkflow:
        @staticmethod
        def build_parser(parser, *, command):
            _ = command
            parser.add_argument("--project-root", required=True)
            parser.add_argument("--save")

        @staticmethod
        def run_main(command, args):
            _ = command
            from reaxkit.core.platform.human_log import current_human_log

            trace = current_human_log()
            assert trace is not None
            trace.completed_step("Read input data", seconds=0.25)
            trace.completed_step("Run analysis", seconds=0.5)
            trace.result("plot", args.save)
            return 0

    spec = SimpleNamespace(name="fake-analysis", module_path="fake.workflow", aliases=())
    cli_module = cast(Any, cli_main)
    module_namespace = vars(cli_module)
    monkeypatch.setitem(
        module_namespace,
        "get_registered_analysis_commands",
        lambda: {"fake-analysis": spec},
    )
    monkeypatch.setitem(module_namespace, "get_registered_generators", lambda: {})
    monkeypatch.setitem(module_namespace, "get_registered_workflows", lambda: {})
    monkeypatch.setitem(module_namespace, "get_registered_commands", lambda **kwargs: {})
    monkeypatch.setitem(module_namespace, "import_module", lambda name: FakeWorkflow)
    output = tmp_path / "plot.png"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reaxkit",
            "fake-analysis",
            "--project-root",
            str(tmp_path),
            "--save",
            str(output),
        ],
    )

    assert cli_main.main() == 0

    log_text = (tmp_path / "logs" / "human_readable.log").read_text(encoding="utf-8")
    assert "REQUEST: ReaxKit CLI command: fake-analysis" in log_text
    assert "command: reaxkit fake-analysis" in log_text
    assert "- Read input data" in log_text
    assert "- Run analysis" in log_text
    assert str(output.resolve()) in log_text

    machine_record = json.loads(
        (tmp_path / "logs" / "machine_readable.jsonl").read_text(encoding="utf-8")
    )
    assert machine_record["request"]["status"] == "success"
    execute_step = machine_record["steps"][0]
    assert execute_step["name"] == "Execute fake-analysis command"
    assert [step["name"] for step in execute_step["steps"]] == [
        "Read input data",
        "Run analysis",
    ]
