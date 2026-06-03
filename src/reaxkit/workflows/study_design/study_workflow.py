"""Study workflow CLI entrypoint.

This module provides the public command entry surface (`build_parser`,
`run_main`) for the `study` command while delegating execution-heavy behavior
to `runtime.py`.

**Usage context**

- CLI routing: Loaded by command registries as the `study` command module.
- Parser ownership: Defines top-level parser description/help wiring.
- Runtime dispatch: Forwards execution to `study_design.runtime`.
"""

from __future__ import annotations

import argparse

from reaxkit.workflows.study_design.cli_help import STUDY_PARSER_DESCRIPTION
from reaxkit.workflows.study_design import runtime as _runtime

STUDY_COMMAND = _runtime.STUDY_COMMAND
ALL_COMMANDS = _runtime.ALL_COMMANDS
ALL_LEGACY_COMMANDS = _runtime.ALL_LEGACY_COMMANDS


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Build and configure the `study` command parser."""
    out = _runtime.build_parser(parser, command=command)
    # Keep the long-form study CLI documentation in the main entry module.
    out.description = STUDY_PARSER_DESCRIPTION
    return out


def run_main(command: str, args: argparse.Namespace) -> int:
    """Run the `study` workflow command."""
    return _runtime.run_main(command, args)

