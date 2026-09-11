"""Tests for the force-field bulk-modulus command name and legacy alias."""

from __future__ import annotations

import argparse

from reaxkit.cli.main import _canonicalize_direct_command
from reaxkit.core.registry.analysis_cli_routing_registry import (
    get_registered_analysis_commands,
)
from reaxkit.workflows.file_tools.ffield_workflow import (
    ALL_COMMANDS,
    ALL_LEGACY_COMMANDS,
    build_parser,
)


def test_bulk_modulus_command_uses_new_canonical_name() -> None:
    routes = get_registered_analysis_commands()

    assert "get_ffield_opt_bulk_modulus" in ALL_COMMANDS
    assert "ffield_opt_bulk_modulus" not in ALL_COMMANDS
    assert "ffield_opt_bulk_modulus" in ALL_LEGACY_COMMANDS
    assert "get_ffield_opt_bulk_modulus" in routes
    assert "ffield_opt_bulk_modulus" not in routes


def test_legacy_bulk_modulus_command_resolves_to_canonical_name() -> None:
    for legacy_name in ("ffield_opt_bulk_modulus", "ffield-opt-bulk-modulus"):
        assert _canonicalize_direct_command(["reaxkit", legacy_name])[1] == (
            "get_ffield_opt_bulk_modulus"
        )

        parser = build_parser(argparse.ArgumentParser(), command=legacy_name)

        args = parser.parse_args([])

        assert args.command == "get_ffield_opt_bulk_modulus"
        assert "reaxkit get_ffield_opt_bulk_modulus" in parser.description
