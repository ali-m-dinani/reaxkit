"""The two CLI help views must describe the same accepted parser actions."""

from __future__ import annotations

import argparse
import ast
from importlib import import_module
import json
from pathlib import Path
import sys

import pytest

from reaxkit.cli.help_flag_registry import FLAG_METADATA
from reaxkit.cli.help_metadata import CATEGORIES, metadata_for_action, metadata_for_flag
from reaxkit.cli.main import _ReaxKitArgumentParser, main


def _help(monkeypatch, capsys, *arguments: str) -> str:
    monkeypatch.setattr(sys, "argv", ["reaxkit", *arguments])
    with pytest.raises(SystemExit) as result:
        main(announce=False)
    assert result.value.code == 0
    return capsys.readouterr().out


def test_registry_roles_are_valid_and_new_flags_require_registration():
    assert all(category in CATEGORIES and visibility in {"short", "full", "internal"}
               for category, visibility in FLAG_METADATA.values())
    parser = _ReaxKitArgumentParser("example")
    action = parser.add_argument("--new-scientific-option")
    with pytest.raises(ValueError, match="Uncategorized CLI option"):
        metadata_for_action(action, parser)


def test_generated_inventory_has_no_unregistered_or_duplicate_flags():
    repo_root = Path(__file__).resolve().parents[2]
    registry_tree = ast.parse((repo_root / "src/reaxkit/cli/help_flag_registry.py").read_text(encoding="utf-8-sig"))
    mapping = next(node.value for node in registry_tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "FLAG_METADATA"
                           for target in node.targets))
    keys = [key.value for key in mapping.keys]
    assert len(keys) == len(set(keys))

    rows = json.loads((repo_root / "cli-help-inventory.json").read_text(encoding="utf-8"))
    assert len(rows) > 4000
    for row in rows:
        flag = next((flag for flag in row["flags"] if flag.startswith("--")), row["flags"][0])
        assert row["category"] == metadata_for_flag(flag, row["parser"])[0]
        if row["required"]:
            assert row["visibility"] == "short"


def test_required_option_is_promoted_to_short_help():
    parser = _ReaxKitArgumentParser("example")
    parser.add_argument("--replication", required=True)
    assert "--replication" in parser.format_help()


def test_context_specific_roles_keep_scientific_selectors_visible():
    assert metadata_for_flag("--yaxis", "reaxkit fort7 get") == ("Scientific choices", "short")
    assert metadata_for_flag("--yaxis", "reaxkit gen-plot")[0] == "Outputs and plots"
    assert metadata_for_flag("--analysis", "reaxkit study") == ("Scientific choices", "short")
    assert metadata_for_flag("--root", "reaxkit study") == ("Outputs and plots", "short")
    assert metadata_for_flag("--src")[0] == "Scientific choices"
    assert metadata_for_flag("--log")[0] == "Diagnostics and compatibility"


@pytest.mark.parametrize("width", [80, 100, 120])
def test_short_table_fits_terminal_and_full_contains_advanced_flags(monkeypatch, capsys, width):
    monkeypatch.setattr(_ReaxKitArgumentParser, "_term_width", staticmethod(lambda: width))
    short = _help(monkeypatch, capsys, "get-hbn-reference-projected-polarity", "-h")
    full = _help(monkeypatch, capsys, "get-hbn-reference-projected-polarity", "--help-all")
    assert max(map(len, short.splitlines())) <= width
    assert max(map(len, full.splitlines())) <= width
    for flag in ("--input", "--engine", "--replication", "--frames", "--periodic",
                 "--charge-source", "--formal-charge", "--reference-frame", "--component",
                 "--local-grouping", "--projection-plane", "--projection-bins",
                 "--plot-kymograph"):
        assert flag in short
    for flag in ("--input-cache", "--no-input-cache", "--execution", "--workers",
                 "--chunk-size", "--project-root", "--xmolout"):
        assert flag in full
        assert flag not in short
    assert "For all options:" in short
    assert all(category in full for category in CATEGORIES)


def test_full_help_wins_over_short_help_in_either_order(monkeypatch, capsys):
    left = _help(monkeypatch, capsys, "get-hbn-reference-projected-polarity", "-h", "--help-all")
    right = _help(monkeypatch, capsys, "get-hbn-reference-projected-polarity", "--help-all", "-h")
    assert left == right
    assert "--no-input-cache" in left
    before_command = _help(monkeypatch, capsys, "--help-all", "--no-stream",
                           "get_hbn_reference_projected_polarity", "-h")
    assert before_command == left


def test_nested_workflow_task_has_both_views(monkeypatch, capsys):
    short = _help(monkeypatch, capsys, "fort7", "get", "-h")
    full = _help(monkeypatch, capsys, "fort7", "get", "--help-all")
    assert "For all options:" in short
    assert "Execution" in full
    assert "--workers" in full
    assert "--workers" not in short


def test_root_generator_and_alias_offer_full_help(monkeypatch, capsys):
    root = _help(monkeypatch, capsys, "--help-all")
    generator = _help(monkeypatch, capsys, "gen_eregime", "--help-all")
    alias = _help(monkeypatch, capsys, "get_hbn_reference_projected_polarity", "--all-flags")
    assert "Commands" in root and "Diagnostics and compatibility" in root
    assert "--help-all" in generator and "Scientific choices" in generator
    assert "--replication" in alias and "--workers" in alias


def test_boolean_pair_occupies_one_full_help_row(monkeypatch, capsys):
    full = _help(monkeypatch, capsys, "get-hbn-reference-projected-polarity", "--help-all")
    assert "--input-cache" in full and "--no-input-cache" in full


def test_help_actions_do_not_parse_required_arguments_or_create_trace(monkeypatch, capsys):
    def reject_trace(*args, **kwargs):
        raise AssertionError("help must exit before creating a workspace trace")

    monkeypatch.setattr(import_module("reaxkit.cli.main"), "HumanReadableRunLog", reject_trace)
    _help(monkeypatch, capsys, "get-hbn-reference-projected-polarity", "--help-all")
