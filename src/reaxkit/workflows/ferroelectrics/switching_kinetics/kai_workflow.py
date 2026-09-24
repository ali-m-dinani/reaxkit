"""CLI workflow for KAI-only fitting."""

from __future__ import annotations

import argparse

from .common import add_common_arguments, run_workflow

COMMAND = "fit-kai-switching"
ALL_COMMANDS = (COMMAND,)
ALL_LEGACY_COMMANDS = ("fit_kai_switching",)


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    if command not in (*ALL_COMMANDS, *ALL_LEGACY_COMMANDS):
        raise KeyError(command)
    parser.set_defaults(command=COMMAND)
    parser.description = "Fit the KAI switching model to a CSV or Excel table."
    return add_common_arguments(parser)


def run_main(command: str, args: argparse.Namespace) -> int:
    if command not in (*ALL_COMMANDS, *ALL_LEGACY_COMMANDS):
        raise KeyError(command)
    return run_workflow(args, ("kai",), command_name=COMMAND)


__all__ = ["ALL_COMMANDS", "ALL_LEGACY_COMMANDS", "COMMAND", "build_parser", "run_main"]
