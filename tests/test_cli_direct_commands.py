from __future__ import annotations

from importlib import import_module

cli_main = import_module("reaxkit.cli.main")


def test_canonicalize_direct_command_alias():
    argv = ["reaxkit", "mean-square-displacement", "--plot"]

    out = cli_main._canonicalize_direct_command(argv)

    assert out[1] == "msd"


def test_canonicalize_direct_command_diffusivity_alias():
    argv = ["reaxkit", "diffusion-coefficient", "--plot"]

    out = cli_main._canonicalize_direct_command(argv)

    assert out[1] == "diffusivity"
