"""Launch the ReaxKit Dash Web UI."""

from __future__ import annotations

import argparse

ALL_COMMANDS = ("gui",)
ALL_LEGACY_COMMANDS = ()


def build_parser(p: argparse.ArgumentParser) -> None:
    """Define CLI arguments for `reaxkit gui`."""
    p.description = "Launch the ReaxKit Web UI."


def run_main(args: argparse.Namespace) -> int:
    """Run the Web UI server."""
    from reaxkit.webui.app import main as webui_main

    webui_main()
    return 0
