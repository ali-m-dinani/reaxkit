"""Minimal, immediate-feedback entry point for the ReaxKit CLI."""

from __future__ import annotations

import sys


def announce_command_start(argv: list[str]) -> None:
    """Confirm receipt of a real command before importing the full CLI."""
    tokens = list(argv[1:])
    if not tokens or any(token in {"-h", "--help", "--help-all", "--all-flags"} for token in tokens):
        return
    command = next((token for token in tokens if not token.startswith("-")), None)
    if command:
        print(
            f"[ReaxKit] Command '{command}' received; starting work...",
            file=sys.stderr,
            flush=True,
        )


def _run_cli() -> int:
    """Import the full dispatcher only after the startup notice is visible."""
    from reaxkit.cli.main import main as dispatch

    return dispatch(announce=False)


def main() -> int:
    """Announce immediately, then hand control to the full CLI dispatcher."""
    announce_command_start(sys.argv)
    return _run_cli()


__all__ = ["main"]
