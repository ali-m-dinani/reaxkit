from __future__ import annotations

import sys
from importlib import import_module

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
