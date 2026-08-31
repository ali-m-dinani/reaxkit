from __future__ import annotations

import json
import sys
from importlib import import_module
from types import SimpleNamespace
from typing import Any, cast

import pytest

cli_main = import_module("reaxkit.cli.main")


def test_canonicalize_direct_command_alias():
    argv = ["reaxkit", "mean-square-displacement", "--plot"]

    out = cli_main._canonicalize_direct_command(argv)

    assert out[1] == "msd"


def test_canonicalize_direct_command_diffusivity_alias():
    argv = ["reaxkit", "diffusion-coefficient", "--plot"]

    out = cli_main._canonicalize_direct_command(argv)

    assert out[1] == "diffusivity"


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
