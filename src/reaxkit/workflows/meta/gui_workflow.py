"""Launch the ReaxKit Dash Web UI.

This module implements CLI workflow orchestration for its command family, including argument parsing, request construction, execution dispatch, and result presentation handoff.

**Usage context**

- Command routing: Resolve CLI aliases and normalized command names.
- Task execution: Build request objects and invoke registered tasks.
- Output handling: Forward results to table, plot, export, or report flows.
"""

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
